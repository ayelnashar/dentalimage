# Bounded prediction updates and cross-dataset shift screening for frozen dental image encoders

Code for the manuscript of the same title by Ayman Elnashar, Norsin Elnashar and Ahmed Elemam,
submitted to the *Journal of Medical Imaging* (SPIE).

The method keeps a pretrained image encoder frozen and works on its embeddings of individual
tooth crops. A temperature-scaled logistic probe gives a reference logit for each condition. A
residual head adds a correction that is clipped to [-δ, δ] in logit space (δ = 1.5), so the final
logit never differs from the reference by more than δ. A Mahalanobis distance from the source
training embeddings, thresholded at a source percentile, decides whether a tooth crop is accepted
or referred back to the clinician.

The repository contains Jupyter notebooks only. It contains no radiographs, tooth crops, cached
embeddings or model weights.

## Notebooks

| Notebook | What it does | Manuscript results |
|---|---|---|
| `01_task_trained_classifiers.ipynb` | Trains DenseNet-121, EfficientNetV2-S and ResNet-50 end to end on DENTEX tooth crops; compares three multi-label heads on frozen features of the best network; fits the bounded update to frozen EfficientNetV2-S features | Table 1, Table 2, Fig. 2, Fig. 6 and the EfficientNetV2-S results |
| `02_bounded_update_single_split.ipynb` | Reference probe, temperature, clipped residual head and Mahalanobis screening for Dental ViT-L and BiomedCLIP on the seed-42 split; operating points, paired bootstrap, calibration comparison and logit-based detectors | Fig. 4, Table 4, bootstrap intervals, temperature/Platt/isotonic comparison, logit-based rows of Table 7 |
| `03_three_splits_and_delta_sweep.ipynb` | The same method on three radiograph-grouped splits, and the sweep over δ; caches the DENTEX embeddings | Table 3, Table 5, Table 6, Fig. 3 |
| `04_tufts_shift_screening.ipynb` | Tufts tooth crops from the segmentation boxes; Mahalanobis distances, refusal at the DENTEX threshold, AUROC, AUPR and FPR95 | Mahalanobis rows of Table 7, Table 8, Fig. 5 |
| `05_adcd_crops.ipynb` | Parses the ADCD external-validation split, crops the Caries, Impacted and Periapical Lesion boxes and caches their Dental ViT-L embeddings | Input to Table 9 |
| `06_threshold_recalibration.ipynb` | Refusal of Tufts and ADCD at the DENTEX validation threshold, and refusal after resetting the threshold from unlabelled target crops | Table 9, ADCD recalibration |
| `07_case_illustration.ipynb` | Per-tooth outputs drawn on one DENTEX radiograph, saved as HTML, PDF and PNG, for Dental ViT-L and optionally DINOv3-L | Fig. 7 |

## 1. Hardware

All results in the manuscript were computed on an Apple Mac Studio (M3 Ultra, 256 GB unified
memory) with the PyTorch Metal Performance Shaders (MPS) backend in 32-bit floating point. Each
notebook selects MPS when available, then CUDA, then the CPU. Extracting the Dental ViT-L
embeddings at 512 × 512 pixels is the slowest step; the embeddings are cached, so later runs are
fast. The residual heads and the end-to-end networks are trained with random initialisation and
dropout, so results obtained on other hardware, or on repeated MPS runs, can differ from the
reported values in the last digits.

## 2. Python environments

Two environments were used.

**Environment A** (Python 3.9.13, PyTorch 2.2.2) for everything that does not load DINOv3-L:

```bash
conda create -n dental-a python=3.9.13
conda activate dental-a
pip install -r requirements.txt
python -m ipykernel install --user --name dental-a
```

**Environment B** (Python 3.12.8, PyTorch 2.12.0, Transformers 5.12.1) for Section 8 of notebook
05 and Section 4b of notebook 07, which load DINOv3-L:

```bash
conda create -n dental-b python=3.12.8
conda activate dental-b
pip install -r requirements-dinov3.txt
python -m ipykernel install --user --name dental-b
```

Select the matching kernel in Jupyter. The notebooks keep a commented-out installation line in
their first code cell; the requirement files above replace it.

## 3. Data

The datasets are not redistributed here. Obtain them from their providers and keep the folder
names below, or edit the paths in the notebooks (Section 5 of this file).

| Dataset | Source | Licence and access |
|---|---|---|
| DENTEX | https://huggingface.co/datasets/ibrahimhamamci/DENTEX | CC BY-NC-SA 4.0 |
| Tufts Dental Database | request form on https://tdd.ece.tufts.edu/ | access granted by the database maintainers at Tufts University, under their terms of use |
| ADCD | https://doi.org/10.5281/zenodo.15487430 | CC BY 4.0; the database usage agreement restricts use to non-commercial research and education |

Expected layout:

```
DENTEX/training_data/
    quadrant-enumeration-disease/      diagnosis annotations and xrays/
Tufts/
    Radiographs/                       panoramic radiographs (.jpg)
    Segmentation/teeth_bbox.json       tooth boxes
    Expert/expert.json                 radiograph-level descriptions
ADCD/panoramic_radiography_yolo_dataset_14_classes/
    <external-validation split>/       images/ and labels/ of the 180 external radiographs
```

DENTEX reuses file names such as `train_564.png` across its sub-folders for different
radiographs. All notebooks read images from `quadrant-enumeration-disease/xrays`, the folder that
belongs to the diagnosis annotations.

## 4. Models

| Encoder | Source | Licence | Notes |
|---|---|---|---|
| Dental ViT-L | https://huggingface.co/Bryceee/Teeth_Visual_Experts_Models | CC BY-NC 4.0 | File `Teeth_Visual_Experts_ViT_L_periapical_images_cls_7diseases.bin`, downloaded on first use. The model definition comes from https://github.com/jeonsworld/ViT-pytorch, which the notebooks clone into the working folder, so `git` and network access are needed on the first run. |
| BiomedCLIP | https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 | MIT | Loaded through `open_clip`. |
| DINOv3-L | https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m | DINOv3 License | Gated. Request access on the model page and accept the licence, then run `huggingface-cli login` once or set the `HF_TOKEN` environment variable. No token is stored in the notebooks. |

## 5. Paths to set

| Notebook | Variables |
|---|---|
| 01, 02, 03 | `DATA_DIR`: the DENTEX `quadrant-enumeration-disease` folder |
| 04 | `DENTEX_DIR`, `TUFTS_DIR` |
| 05 | `ADCD_ROOT`, and `ADCD_DIR`: replace `EXTERNAL_VALIDATION_FOLDER` with the name of the external-validation split folder |
| 06 | `DATA_DIR`, `TUFTS_DIR`, `ADCD_CACHE` (the `_cache` folder written by notebook 05) |
| 07 | `DATA_DIR`, `MANUAL_IMAGE` (the radiograph to draw), `MANUAL_SAVE_DIR` (where reports are saved) |

## 6. Run order

1. `03_three_splits_and_delta_sweep.ipynb`. It writes the DENTEX embedding cache that notebook 04
   reads.
2. `04_tufts_shift_screening.ipynb`. It writes the Tufts tooth-crop embeddings that Section 16 of
   notebook 02 reads.
3. `01_task_trained_classifiers.ipynb` and `02_bounded_update_single_split.ipynb`, in any order.
   Run each from top to bottom: the additional analyses at the end of each notebook use the
   models fitted earlier in the same kernel.
4. `05_adcd_crops.ipynb`, then `06_threshold_recalibration.ipynb`.
5. `07_case_illustration.ipynb`.

Notebooks 04, 05 and 06 each draw the mapped or parsed boxes on one radiograph before any crop is
embedded. Check that the boxes sit on teeth (Tufts) or on the annotated conditions (ADCD) before
continuing.

Random seeds: radiograph-grouped splits use seed 42, and seeds 42, 43 and 44 in notebook 03. The
bootstrap in notebook 02 uses `numpy.random.default_rng(0)`. The threshold recalibration in
notebook 06 uses seeds 0 to 49 for its 50 random draws.

## 7. Outputs

Results are written next to the data:

| Notebook | Location | Files |
|---|---|---|
| 01 | `DATA_DIR/_results/` | `summary_seed42.json`, reliability diagram PNG, `train_564_efficientnet_RAW.png`, `train_564_efficientnet_ENVELOPED.png`, checkpoints |
| 02 | `DATA_DIR/_results_option1/` | `option1_summary_seed42.json`, `ood_refusal_curves.png`; the additional analyses print their tables |
| 03 | `DATA_DIR/_results_option1/` | `option1_crn_deltasweep.json`, `delta_sweep.png`, `emb_cache/` |
| 04 | `DENTEX_DIR/_results_option1/` | `option1_tufts_ood_toothcrops.json`, `tufts_ood_distance.png`, Tufts embeddings in `emb_cache/` |
| 05 | `ADCD_ROOT/_cache/` | `emb_adcd_dental_vitl.npy`, `lab_adcd.npy` |
| 06 | `DATA_DIR/_results_recal/` | `deployability.png`, `refusal_vs_quality.png`, `adcd_deployability.png`, `cache/` |
| 07 | `MANUAL_SAVE_DIR` | `findings_<image>.html`, `findings_<image>.pdf`, `annotated_<image>.png` |

## 8. Licence

The code is released under the MIT License (see `LICENSE`). The datasets and model checkpoints
remain under their own licences, listed above; several of them restrict use to non-commercial
research.
