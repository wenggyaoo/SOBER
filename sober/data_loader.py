"""Load, align, and normalize study/region cell tables."""
import csv as _csv_mod
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from torch.utils.data import Dataset

from .cell_type_normalize import canonical_raw_cell_type_label

_CELLTYPE_VERSION_RE = re.compile(r"\.cell_types?\.([^.]+)\.csv$", re.IGNORECASE)


def _read_cell_table(path):
    columns = pd.read_csv(path, nrows=0).columns
    identifiers = [name for name in columns
                   if name.lower() in ("cell_id", "cellid", "id", "cell")
                   or ("cell" in name.lower() and "id" in name.lower())]
    return pd.read_csv(path, converters={name: str for name in identifiers})


def normalize_biomarker_expression(bm_exp_df):
    """Apply per-region arcsinh scaling, divide by marker standard deviation, and clip.

    The marker mean is not subtracted; zero-variance markers use a scale of one.
    """
    assert 'CELL_ID' not in bm_exp_df.columns
    denom = 5 * np.quantile(bm_exp_df, 0.2, axis=0) + 1e-5
    bm_exp_df = np.arcsinh(bm_exp_df / np.where(denom > 0, denom, 1.0))
    std = bm_exp_df.std(0)
    std = np.where(std > 1e-8, std, 1.0)
    bm_exp_df = bm_exp_df / std
    bm_exp_df = np.clip(np.nan_to_num(bm_exp_df, nan=0.0, posinf=5.0, neginf=0.0), 0, 5)

    return bm_exp_df


class SpatialBiomarkerDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.regions = []
        self.all_biomarkers = set()
        self.all_cell_types = set()
        self.partial_celltype_alignment_events = []
        self.biomarker_rename = config.biomarker_aliases()
        self._study_allowed_versions = self._load_study_celltype_versions()
        self.skipped_regions = []
        for study in sorted(Path(data_path).iterdir()):
            if not study.is_dir() or study.name.startswith("."):
                continue
            for folder in sorted(study.iterdir()):
                if not folder.is_dir() or folder.name.startswith("."):
                    continue
                region = self.load_region(folder, study.name)
                if region["num_cells"] < config.min_cells_per_region:
                    self.skipped_regions.append({"study_name": study.name, "region_name": folder.name,
                                                 "num_cells": region["num_cells"]})
                    continue
                self.regions.append(region)
                self.all_biomarkers.update(region["biomarkers"])
                self.all_cell_types.update(label for label in region["cell_types"] if label)
        if not self.regions:
            raise ValueError("No eligible regions loaded; check the study/region layout and min_cells_per_region")
        self.all_biomarkers = sorted(self.all_biomarkers)
        self.build_cell_type_vocab()
        print(f"Loaded {len(self.regions)} regions; skipped {len(self.skipped_regions)} small regions")

    def load_region(self, folder, study):
        position_path = self._resolve_region_csv(
            str(folder), self.config.position_filename, "cell_data",
            aliases=["coordinates.csv"], stem_aliases=["coordinates"])
        position = _read_cell_table(position_path)
        if len(position) < self.config.min_cells_per_region:
            return {"study_name": study, "region_name": folder.name, "num_cells": len(position)}
        expression_path = self._resolve_region_csv(str(folder), self.config.biomarker_filename, "expression")
        expression = _read_cell_table(expression_path)
        pos_cols = self.identify_position_columns(position)
        bio_cols = self.identify_biomarker_columns(expression)
        if not pos_cols or not bio_cols.get("cell_id"):
            raise ValueError(f"Missing coordinate or cell ID columns in {folder}")
        expression = self._align_frame_to_position_ids(position, expression, bio_cols["cell_id"], "expression")
        markers = [name for name in bio_cols["biomarkers"]
                   if self.biomarker_rename.get(name, name) != "EMPTY"]
        canonical = [self.biomarker_rename.get(name, name) for name in markers]
        if len(markers) < 2:
            raise ValueError(f"At least two numeric biomarkers are required in {folder}")
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"Marker aliases create duplicate columns in {folder}; merge or rename them before training")
        raw = expression[markers].to_numpy(dtype=float)
        coordinates = position[[pos_cols["x"], pos_cols["y"]]].to_numpy(dtype=float)
        if not np.isfinite(raw).all() or not np.isfinite(coordinates).all():
            raise ValueError(f"Non-finite coordinates or intensities in {folder}")
        if (raw < 0).any():
            raise ValueError(f"Expected non-negative raw intensities in {folder}")
        cell_types = []
        celltype_path = None
        if self.config.use_cell_types:
            celltype_path = self._resolve_celltype_csv(str(folder), study, allow_unversioned_cell_types=True)
            if celltype_path:
                labels = _read_cell_table(celltype_path)
                label_cols = self.identify_celltype_columns(labels)
                if not label_cols:
                    raise ValueError(f"Missing cell ID/type columns in {celltype_path}")
                labels = self._align_frame_to_position_ids(
                    position, labels, label_cols["cell_id"], "cell type", allow_missing_ids=True,
                    study_name=study, region_name=folder.name)
                cell_types = [canonical_raw_cell_type_label(v) for v in labels[label_cols["cell_type"]]]
        return {"study_name": study, "region_name": folder.name,
                "cell_ids": position[pos_cols["cell_id"]].tolist(), "coordinates": coordinates,
                "intensities": np.asarray(normalize_biomarker_expression(expression[markers].copy())),
                "biomarkers": canonical, "cell_types": cell_types, "num_cells": len(position),
                "cell_type_file_used": Path(celltype_path).name if celltype_path else None,
                "kdtree": KDTree(coordinates)}

    def _load_study_celltype_versions(self) -> Dict[str, List[str]]:
        """Read annotation versions in priority order for each study."""
        path = getattr(self.config, "study_celltype_version_file", None)
        if not path:
            return {}
        result: Dict[str, List[str]] = {}
        with io.StringIO(self.config.read_asset("study_celltype_version_file")) as f:
            reader = _csv_mod.DictReader(f, delimiter="\t")
            for row in reader:
                study = (row.get("study_id") or "").strip()
                ver_str = (row.get("cell_type_versions") or "").strip()
                if study and ver_str:
                    result[study] = [v.strip() for v in ver_str.split(",") if v.strip()]
        if result:
            print("Loaded cell-type version table: %d studies from %s" % (len(result), path))
        return result

    def _resolve_region_csv(self,
                            region_path: str,
                            exact_name: str,
                            stem: str,
                            aliases: Optional[List[str]] = None,
                            stem_aliases: Optional[List[str]] = None) -> str:
        """Prefer an exact filename; otherwise choose the last matching name in lexical order."""
        for name in [exact_name] + list(aliases or []):
            exact_path = os.path.join(region_path, name)
            if os.path.exists(exact_path):
                return exact_path
        stems = [stem] + list(stem_aliases or [])
        candidates = [
            f for f in os.listdir(region_path)
            if f.endswith(".csv") and any(s.lower() in f.lower() for s in stems)
        ]
        if not candidates:
            raise FileNotFoundError(f"Missing required file: {exact_path} (no *{stem}*.csv found)")
        candidates.sort()
        return os.path.join(region_path, candidates[-1])

    def _resolve_celltype_csv(
            self,
            region_path: str,
            study_id: str,
            allow_unversioned_cell_types: bool = False) -> Optional[str]:
        """Prefer the study's registered annotation version; allow standard files when requested."""
        if not self._study_allowed_versions and not allow_unversioned_cell_types:
            return None

        allowed = self._study_allowed_versions.get(study_id)
        if not allowed and not allow_unversioned_cell_types:
            return None

        candidates = [
            f for f in os.listdir(region_path)
            if f.endswith(".csv") and ("cell_type" in f.lower() or "celltype" in f.lower())
        ]
        if not candidates:
            return None

        allowed_versions = allowed or []
        allowed_set = set(allowed_versions)
        versioned = []
        for f in candidates:
            m = _CELLTYPE_VERSION_RE.search(f)
            if m and m.group(1) in allowed_set:
                versioned.append((allowed_versions.index(m.group(1)), f))
        if versioned:
            versioned.sort()
            return os.path.join(region_path, versioned[0][1])
        if allow_unversioned_cell_types:
            preferred = [
                getattr(self.config, "celltype_filename", "cell_type.csv"),
                "cell_types.csv",
                "cell_type.csv",
            ]
            for name in preferred:
                if name in candidates:
                    return os.path.join(region_path, name)
            candidates.sort()
            return os.path.join(region_path, candidates[0])
        return None

    def _align_frame_to_position_ids(
            self,
            position_df: pd.DataFrame,
            other_df: pd.DataFrame,
            other_cell_id_col: str,
            label: str,
            allow_missing_ids: bool = False,
            study_name: str = "",
            region_name: str = "") -> pd.DataFrame:
        """Align rows by cell ID, optionally leaving missing annotations as NaN."""
        position_cols = self.identify_position_columns(position_df)
        if not position_cols:
            raise ValueError("Could not identify required position columns before %s alignment" % label)
        pos_cell_id_col = position_cols["cell_id"]
        pos_ids = position_df[pos_cell_id_col].tolist()
        other_ids = other_df[other_cell_id_col].tolist()
        if position_df[pos_cell_id_col].isna().any() or other_df[other_cell_id_col].isna().any():
            raise ValueError("Cell IDs must be non-empty")
        if any(frame[column].astype(str).str.strip().eq("").any()
               for frame, column in ((position_df, pos_cell_id_col), (other_df, other_cell_id_col))):
            raise ValueError("Cell IDs must be non-empty")
        if not position_df[pos_cell_id_col].is_unique:
            n_dup = int(position_df[pos_cell_id_col].duplicated().sum())
            raise ValueError("coordinate table has duplicate cell_id rows (%d duplicates)" % n_dup)
        if not other_df[other_cell_id_col].is_unique:
            n_dup = int(other_df[other_cell_id_col].duplicated().sum())
            raise ValueError("%s table has duplicate cell_id rows (%d duplicates)" % (label, n_dup))
        if pos_ids == other_ids:
            return other_df
        pos_set = set(pos_ids)
        other_set = set(other_ids)
        if pos_set != other_set:
            missing = len(pos_set - other_set)
            extra = len(other_set - pos_set)
            if allow_missing_ids:
                self.partial_celltype_alignment_events.append({
                    "study_name": str(study_name),
                    "region_name": str(region_name),
                    "label": str(label),
                    "missing": int(missing),
                    "extra": int(extra),
                })
                aligned = other_df.set_index(other_cell_id_col).reindex(pos_ids).reset_index()
                aligned[other_cell_id_col] = pos_ids
                return aligned
            raise ValueError(
                "%s table cell_id set mismatch vs coordinates (missing=%d, extra=%d)"
                % (label, missing, extra)
            )
        return other_df.set_index(other_cell_id_col).loc[pos_ids].reset_index()

    def identify_celltype_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        cell_id_candidates = ['cell_id', 'cellid', 'id', 'cell', 'Cell_ID', 'CellID', 'CELL_ID']
        celltype_candidates = [
            'cell_type', 'celltype', 'type', 'Cell_Type', 'CellType',
            'cell_class', 'class', 'ANNOTATION_LABEL', 'annotation_label',
            'label', 'Label', 'LABEL'
        ]

        cell_id_col = None
        celltype_col = None

        for col in df.columns:
            if col in cell_id_candidates or any(
                    candidate.lower() in col.lower() for candidate in ['cell_id', 'cellid']):
                cell_id_col = col
                break

        for col in df.columns:
            if col in celltype_candidates or any(
                    candidate.lower() in col.lower() for candidate in [
                        'cell_type', 'celltype', 'type', 'annotation', 'label'
                    ]):
                celltype_col = col
                break

        if not cell_id_col or not celltype_col:
            return {}

        return {'cell_id': cell_id_col, 'cell_type': celltype_col}

    def identify_position_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        cell_id_candidates = ['cell_id', 'cellid', 'id', 'cell', 'Cell_ID', 'CellID']
        x_candidates = ['x', 'X', 'x_coord', 'X_coord', 'pos_x', 'centroid_x']
        y_candidates = ['y', 'Y', 'y_coord', 'Y_coord', 'pos_y', 'centroid_y']

        cell_id_col = None
        x_col = None
        y_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col in cell_id_candidates or 'cell' in col_lower and 'id' in col_lower:
                cell_id_col = col
            elif col in x_candidates or ('x' in col_lower and any(kw in col_lower for kw in ['coord', 'pos'])):
                x_col = col
            elif col in y_candidates or ('y' in col_lower and any(kw in col_lower for kw in ['coord', 'pos'])):
                y_col = col

        if not all([cell_id_col, x_col, y_col]):
            return {}

        return {'cell_id': cell_id_col, 'x': x_col, 'y': y_col}

    def identify_biomarker_columns(self, df: pd.DataFrame) -> Dict:
        cell_id_candidates = ['cell_id', 'cellid', 'id', 'cell', 'Cell_ID', 'CellID']

        cell_id_col = None
        biomarker_cols = []

        for col in df.columns:
            col_lower = col.lower()
            if col in cell_id_candidates or ('cell' in col_lower and 'id' in col_lower):
                cell_id_col = col
                break

        for col in df.columns:
            if col != cell_id_col and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                biomarker_cols.append(col)

        return {'cell_id': cell_id_col, 'biomarkers': biomarker_cols}

    def build_cell_type_vocab(self):
        """Reserve vocabulary index 0 for unknown cell types."""
        unique_cell_types = sorted(self.all_cell_types)

        self.cell_type_to_idx = {"unknown": 0}

        for idx, cell_type in enumerate(unique_cell_types, start=1):
            if cell_type != "unknown":
                self.cell_type_to_idx[cell_type] = idx

        self.idx_to_cell_type = {idx: cell_type for cell_type, idx in self.cell_type_to_idx.items()}

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        return self.regions[idx]
