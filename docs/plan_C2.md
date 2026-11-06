# Plan C2 — Baseline nnU-Net training, prediction and scoring

> Saved verbatim from the task prompt so later sessions can see the plan.

Preprocessing of the 600-case subset is done (kernel `nitikaarajesh/brats-01-preprocess`).
Now do training, prediction and scoring in ONE Kaggle GPU run, to save round trips.
Explain each part in plain words.

**Evaluation targets:** evaluate on the GLI internal test set (190 cases) AND the
Africa external test set (95 cases), each with all 4 sequences and again with T1c
missing. Skip the val subset: nnU-Net already scores it at the end of training.

Create `kaggle/02_train_eval/` with `02_train_eval.ipynb` and `kernel-metadata.json`:
private, GPU ON (P100 or T4), internet ON, dataset source
`nitikaarajesh/brats2023-gli-nnunet`, and kernel source
`nitikaarajesh/brats-01-preprocess` (its output is the preprocessed data).
Push it with `kaggle kernels push`.

The notebook should:

**Setup.** `pip install nnunetv2`. Copy the preprocessed folder from the kernel-source
input to `/tmp/nnUNet_preprocessed`, because it must be writable. Rebuild the small
symlink raw folder for the 600 subset cases as in notebook 01 (nnU-Net needs its
`dataset.json`). Set `nnUNet_raw`, `nnUNet_preprocessed=/tmp/nnUNet_preprocessed` and
`nnUNet_results=/kaggle/working/nnUNet_results`.

**Train.** `nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_50epochs`. Print the time
of the first 2 epochs. The whole notebook must finish within Kaggle's 12-hour limit, so
if training plus prediction won't fit, stop and tell me rather than guessing. Save the
training log and `progress.png`.

**Predict** with `nnUNetv2_predict -chk checkpoint_final.pth --disable_tta` on four sets:

(a) GLI test, 190 cases;
(b) Africa, 95 cases;
(c) GLI test with T1c missing;
(d) Africa with T1c missing.

Build each input folder from the dataset mount with `.nii` file names, and handle the
nested GLI image folders the same way as notebook 01. For c and d, replace channel
`_0001` (T1c) with a zero image that has the same header. Use `--save_probabilities`,
then for each case immediately compute the uncertainty: the mean entropy inside the
predicted whole-tumour region. After that, delete the probability files to keep the disk
small.

**Score** every prediction against its label: Dice and HD95 for whole tumour (labels
1+2+3), tumour core (1+3) and enhancing tumour (3). If both the label and the prediction
have no voxels for a region, the Dice is 1. If only one is empty, the Dice is 0. Save one
CSV with the columns: `case_id, set (gli/africa), condition (full/no_t1c), dice_wt,
dice_tc, dice_et, hd95_wt, hd95_tc, hd95_et, uncertainty`.

Keep `/kaggle/working` small. Keep only the results CSV, the training log, `progress.png`,
`checkpoint_final.pth` and the predicted masks as `.nii.gz`.

Track the run with `kaggle kernels status`. When it finishes, download the outputs into a
git-ignored `results_raw/` folder. Copy only the CSV, `progress.png` and the training log
into `results/` in the repo. Commit them, together with `kaggle/02_train_eval/` and
`docs/plan_C2.md`, with the message "Baseline nnU-Net results".

Then report:

- the mean Dice (± SD) for each region, for GLI full, Africa full, GLI without T1c and
  Africa without T1c;
- the mean HD95 for the same groups;
- the correlation between uncertainty and whole-tumour Dice, for each set.
