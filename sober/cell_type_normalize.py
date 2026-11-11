"""Map annotations to coarse/granular cell-type labels.

Missing coarse labels are excluded from supervision. Missing granular labels
retain coarse supervision only.
"""
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HierarchyLabel = Optional[Tuple[str, str]]  # Empty granular labels retain coarse supervision.


def canonical_raw_cell_type_label(val: Any) -> str:
    """Normalize scalar annotations to strings; represent missing values as an empty string."""
    if val is None or val is pd.NA:
        return ""
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return ""
        if np.isfinite(val) and float(val) == int(val):
            return str(int(val))
        return str(val).strip()
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def load_cell_type_mapping_csv(path: str) -> Dict[str, HierarchyLabel]:
    """Read raw label -> (coarse, granular); missing coarse labels map to None."""
    if not path or not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    col_orig = None
    col_coarse = None
    col_gran = None
    for c in df.columns:
        cl = str(c).strip().lower().replace(" ", "_")
        if cl == "original_name":
            col_orig = c
        elif cl == "coarse_cell_type":
            col_coarse = c
        elif cl == "granular_cell_type":
            col_gran = c
    if not col_orig or not col_coarse:
        raise ValueError("cell_type_mappings.csv must contain Original_name and Coarse_cell_type columns")

    mapping: Dict[str, HierarchyLabel] = {}

    def _is_na_cell(val: Any) -> bool:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return True
        t = str(val).strip().upper()
        return t in ("", "NA", "N/A", "NAN", "NONE")

    for _, row in df.iterrows():
        orig = row[col_orig]
        if pd.isna(orig):
            continue
        key = canonical_raw_cell_type_label(orig)
        if not key:
            continue
        coarse_raw = row[col_coarse]
        if _is_na_cell(coarse_raw):
            mapping[key] = None
            continue
        coarse = str(coarse_raw).strip()
        gran = ""
        if col_gran is not None:
            g = row[col_gran]
            if not _is_na_cell(g):
                gran = str(g).strip()
        mapping[key] = (coarse, gran)

    return mapping


def build_mapping_casefold(mapping: Dict[str, HierarchyLabel]) -> Dict[str, HierarchyLabel]:
    """Build a case-insensitive lookup, retaining the first key on collisions."""
    out: Dict[str, HierarchyLabel] = {}
    for k, v in mapping.items():
        ck = k.strip().casefold()
        if ck not in out:
            out[ck] = v
    return out


def _raw_label_variants(raw: Any) -> List[str]:
    """Try the original label and an integer form for numeric labels such as "1.0"."""
    if raw is None or _is_nan(raw):
        return []
    if isinstance(raw, (int, np.integer)):
        return [str(int(raw))]
    if isinstance(raw, float) and not np.isnan(raw):
        if raw == int(raw):
            return [str(int(raw)), str(raw)]
        return [str(raw).strip()]
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    out = [s]
    try:
        f = float(s)
        if not np.isnan(f) and f == int(f):
            out.append(str(int(f)))
    except ValueError:
        pass
    return list(dict.fromkeys(out))


def hierarchy_to_display_label(h: HierarchyLabel) -> str:
    """Format a label as coarse, coarse|granular, or unknown."""
    if h is None:
        return "unknown"
    c, g = h
    if not g:
        return c
    return "%s|%s" % (c, g)


def raw_to_hierarchy(
    raw: Any,
    mapping: Dict[str, HierarchyLabel],
    mapping_cf: Optional[Dict[str, HierarchyLabel]] = None,
) -> HierarchyLabel:
    """Try exact label variants, then case-insensitive variants; return None if unmapped."""
    variants = _raw_label_variants(raw)
    if not variants:
        return None

    for s in variants:
        if s in mapping:
            return mapping[s]

    if mapping_cf is not None:
        for s in variants:
            ck = s.casefold()
            if ck in mapping_cf:
                return mapping_cf[ck]

    return None


def apply_cell_type_normalization(
    dataset,
    mapping_csv_path: Optional[str] = None,
    verbose: bool = True,
):
    """Attach hierarchy labels to each region and rebuild the display-label vocabulary."""
    if not getattr(dataset.config, "use_cell_types", True):
        return 0, 0

    cfg_path = mapping_csv_path
    if cfg_path is None:
        cfg_path = getattr(dataset.config, "cell_type_mapping_file", None)

    mapping: Dict[str, HierarchyLabel] = {}
    if cfg_path and os.path.isfile(cfg_path):
        mapping = load_cell_type_mapping_csv(cfg_path)
        if verbose:
            print("Loaded cell type mapping: %d rows from %s" % (len(mapping), os.path.abspath(cfg_path)))
    elif verbose and cfg_path:
        print("Warning: cell_type_mapping_file not found (%s); labels stay unmapped" % cfg_path)
    n_mapped = 0
    n_csv_na = 0       # Explicitly excluded by the mapping.
    n_unmapped = 0     # Annotated but absent from the mapping.
    n_blank_annotation = 0
    n_cells_with_labels = sum(len(r.get("cell_types") or []) for r in dataset.regions)
    n_before = len(dataset.all_cell_types)
    unmapped_examples: Dict[str, int] = {}

    mapping_cf = build_mapping_casefold(mapping) if mapping else {}

    def _label_in_mapping(raw: Any) -> bool:
        for s in _raw_label_variants(raw):
            if s in mapping:
                return True
        for s in _raw_label_variants(raw):
            if s.casefold() in mapping_cf:
                return True
        return False

    dataset.all_cell_types = set()
    for r in dataset.regions:
        raw_list = r.get("cell_types") or []
        hier: List[HierarchyLabel] = []
        display: List[str] = []
        for raw in raw_list:
            variants = _raw_label_variants(raw)
            is_blank = len(variants) == 0

            h = raw_to_hierarchy(raw, mapping, mapping_cf=mapping_cf or None)
            hier.append(h)

            in_csv = _label_in_mapping(raw)
            s_repr = variants[0] if variants else ""

            if is_blank:
                n_blank_annotation += 1
            elif h is not None:
                n_mapped += 1
            else:
                if in_csv:
                    n_csv_na += 1
                else:
                    n_unmapped += 1
                    unmapped_examples[s_repr] = unmapped_examples.get(s_repr, 0) + 1

            display.append(hierarchy_to_display_label(h))
        r["cell_type_hierarchy"] = hier
        r["cell_types"] = display
        dataset.all_cell_types.update(display)

    dataset.build_cell_type_vocab()
    n_after = len(dataset.all_cell_types)
    total_bucket = n_mapped + n_csv_na + n_unmapped + n_blank_annotation
    if verbose:
        print(
            "Cell type normalization: %d unique canonical raw strings -> %d display labels"
            % (n_before, n_after)
        )
        print(
            "  (unique = non-empty raw strings only; blank/NaN in annotation column are not counted.)"
        )
        print(
            "  cells (one list slot per cell in region): %d" % n_cells_with_labels
        )
        print(
            "  blank annotation cell  = %d cells (no text after canonical; not 'missing from CSV')"
            % n_blank_annotation
        )
        print(
            "  mapped (CSV hit)      = %d cells" % n_mapped
        )
        print(
            "  NA (in CSV, excluded) = %d cells" % n_csv_na
        )
        print(
            "  unmapped (non-blank raw, not in CSV) = %d cells" % n_unmapped
        )
        if total_bucket != n_cells_with_labels:
            print(
                "  WARNING: sum of buckets (%d) != cells with labels (%d) (logic bug?)"
                % (total_bucket, n_cells_with_labels)
            )
        if unmapped_examples:
            top = sorted(unmapped_examples.items(), key=lambda x: -x[1])[:30]
            print("  Unique unmapped raw labels: %d" % len(unmapped_examples))
            print("  Top unmapped raw labels: %s"
                  % ", ".join("%s (%d)" % (k, v) for k, v in top))
    return n_before, n_after
