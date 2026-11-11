"""Patient filtering and grouped training/internal-validation splits."""
import os

import numpy as np
import pandas as pd


def _clean_identifier(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def canonical_patient_id(value):
    """Normalize patient IDs across pickle/CSV numeric-vs-string representations."""
    value = _clean_identifier(value)
    if value is None:
        return None
    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return value


def filter_dataset_by_patient_metadata(dataset, metadata_path):
    """Keep regions with a unique, non-empty ACQUISITION_ID -> PATIENT_ID mapping."""
    metadata_path = os.path.abspath(metadata_path)
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError("Patient metadata not found: %s" % metadata_path)
    suffix = os.path.splitext(metadata_path)[1].lower()
    if suffix in (".csv", ".tsv"):
        metadata = pd.read_csv(metadata_path, sep="\t" if suffix == ".tsv" else ",", dtype=str)
    elif suffix in (".pkl", ".pickle"):
        metadata = pd.read_pickle(metadata_path)
    else:
        raise ValueError("Patient metadata must be CSV, TSV, or a trusted pickle")
    required = {"ACQUISITION_ID", "PATIENT_ID"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError("Patient metadata is missing column(s): %s" % ", ".join(sorted(missing)))

    acquisition_to_patient = {}
    ambiguous_ids = set()
    metadata_missing_patient = set()
    for acquisition_id, patient_id in metadata[["ACQUISITION_ID", "PATIENT_ID"]].itertuples(index=False):
        acquisition_id = _clean_identifier(acquisition_id)
        patient_id = _clean_identifier(patient_id)
        if acquisition_id is None:
            continue
        if patient_id is None:
            metadata_missing_patient.add(acquisition_id)
            continue
        previous = acquisition_to_patient.get(acquisition_id)
        if previous is not None and previous != patient_id:
            ambiguous_ids.add(acquisition_id)
        else:
            acquisition_to_patient[acquisition_id] = patient_id
    for acquisition_id in ambiguous_ids:
        acquisition_to_patient.pop(acquisition_id, None)

    kept_regions = []
    excluded = {"unmatched": [], "missing_patient_id": [], "ambiguous_patient_id": []}
    for region in dataset.regions:
        region_id = _clean_identifier(region.get("region_name"))
        if region_id in ambiguous_ids:
            excluded["ambiguous_patient_id"].append(region_id)
        elif region_id in acquisition_to_patient:
            region["patient_id"] = acquisition_to_patient[region_id]
            kept_regions.append(region)
        elif region_id in metadata_missing_patient:
            excluded["missing_patient_id"].append(region_id)
        else:
            excluded["unmatched"].append(region_id)

    n_loaded = len(dataset.regions)
    if not kept_regions:
        raise ValueError("No loaded regions matched a non-empty PATIENT_ID in %s" % metadata_path)
    dataset.regions = kept_regions
    dataset.all_biomarkers = sorted({b for region in kept_regions for b in region.get("biomarkers", [])})
    dataset.all_cell_types = {
        cell_type for region in kept_regions
        for cell_type in region.get("cell_types", []) if cell_type
    }
    if getattr(dataset.config, "use_cell_types", False):
        dataset.build_cell_type_vocab()

    return {
        "metadata_path": metadata_path,
        "matching_columns": {"region": "region_name", "acquisition": "ACQUISITION_ID", "patient": "PATIENT_ID"},
        "n_regions_loaded": n_loaded,
        "n_regions_kept": len(kept_regions),
        "n_patients_kept": len({region["patient_id"] for region in kept_regions}),
        "n_regions_excluded_unmatched": len(excluded["unmatched"]),
        "n_regions_excluded_missing_patient": len(excluded["missing_patient_id"]),
        "n_regions_excluded_ambiguous_mapping": len(excluded["ambiguous_patient_id"]),
        "excluded_region_ids": {
            key: sorted(value for value in values if value is not None)
            for key, values in excluded.items()
        },
    }


def make_patient_split(dataset, train_ratio=0.8, split_seed=42):
    """Split regions by patient; each patient is assigned to one partition."""
    patient_to_indices = {}
    for index, region in enumerate(dataset.regions):
        patient_id = _clean_identifier(region.get("patient_id"))
        if patient_id is None:
            raise ValueError("Patient split requires every retained region to have patient_id")
        patient_to_indices.setdefault(patient_id, []).append(index)
    patients = sorted(patient_to_indices)
    if len(patients) < 2:
        raise ValueError("Patient split requires at least two distinct patients")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 for patient split")
    rng = np.random.RandomState(split_seed)
    rng.shuffle(patients)
    n_train = min(len(patients) - 1, max(1, int(len(patients) * train_ratio)))
    train_patients = set(patients[:n_train])
    train_indices = [i for p, indices in patient_to_indices.items() if p in train_patients for i in indices]
    validation_indices = [i for p, indices in patient_to_indices.items() if p not in train_patients for i in indices]
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices, sorted(train_patients), sorted(set(patients) - train_patients)
