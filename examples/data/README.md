# Example data

Three complete tissue regions are included for the [data loading tutorial](../../tutorials/data_loading.ipynb).

| Collection | Role in SOBER | Platform | Region | Cells | Marker columns |
| --- | --- | --- | --- | ---: | ---: |
| UPMC-HNC (`s240`) | Internal training | CODEX | `UPMC_c001_v001_r001_reg039` | 1,670 | 41 |
| Damond et al., pancreas | External | IMC | `N43` | 1,500 | 35 |
| Meyer et al., breast cancer | External | IMC | `ZTMA249.1_ZTMA249_ID_48_AX8Y6_30` | 1,500 | 36 |

**Internal** refers to the SOBER development atlas. The selected `s240` region appears in `train_regions`, and not in `validation_regions`, in the recorded patient split (`training_contrastive_patient_300/train_test_split.json`, seed 42). It is a training example, not a held-out evaluation sample. **External** identifies the two collections recorded outside that atlas; neither example region appears in the saved split. These three regions illustrate data loading and do not form a benchmark split.

Each folder contains `cell_data.csv`, `expression.csv`, and `cell_type.csv`. All supplied cells are retained. The external regions were selected by proximity to 1,500 cells, breaking ties by region name; the internal region was selected as a small, complete training example.

## Internal example

The UPMC-HNC region comes from `three_dataset_demo/s240/UPMC_c001_v001_r001_reg039` in the research repository. The project crosswalk identifies `s240` as the UPMC head-and-neck cancer CODEX cohort described by [Wu et al. (2022)](https://doi.org/10.1038/s41551-022-00951-w).

Expression values and coordinates are copied unchanged. Cell-type strings come from `UPMC_c001_v001_r001_reg039.cell_types.95-5-4.csv`, the annotation version selected for `s240` in `study_celltype_final.tsv`; the alternative local label files are not used. Headers are standardized and numeric label codes are omitted. All 41 input channels, including DAPI, are retained by the current loader.

Coordinates remain in their native source units. Physical calibration has not been verified for this local export, so the tutorial does not label its axes in micrometres. The local CSVs provide no redistribution license; the external-data licenses below do not cover this example.

## External sources and attribution

The two external examples were extracted from `sce_full.rds` SingleCellExperiment objects distributed by [imcdatasets](https://github.com/BodenmillerGroup/imcdatasets):

- Damond N. et al. *A Map of Human Type 1 Diabetes Progression by Imaging Mass Cytometry*. Cell Metabolism (2019). [Article](https://doi.org/10.1016/j.cmet.2018.11.014); [imcdatasets deposit](https://doi.org/10.5281/zenodo.20623726), credited to Nathan Steenbuck and Nicolas Damond; [original data deposit](https://doi.org/10.17632/cydmwsfztj.2).
- Meyer L. et al. *A stratification system for breast cancer based on basoluminal tumor cells and spatial tumor architecture*. Cancer Cell (2025). [Article](https://doi.org/10.1016/j.ccell.2025.06.019); [imcdatasets deposit](https://doi.org/10.5281/zenodo.20541924), credited to Nathan Steenbuck and Lasse Meyer.

Both source deposits specify [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The two external examples retain that license and attribution; this notice does not license the internal example or SOBER code.
