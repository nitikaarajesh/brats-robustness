"""Index all BraTS cases and make patient-level train/val/test splits."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(r"C:\data\brats")
SOURCES = {"GLI": DATA / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
           "SSA": DATA / "BraTS-Africa" / "95_Glioma"}
MODS = ["t1n", "t1c", "t2w", "t2f", "seg"]
OUT = Path(__file__).resolve().parents[1] / "splits"

rows = []
for source, root in SOURCES.items():
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        files = {f.name.split("-")[-1].replace(".nii.gz", ""): f for f in case.rglob("*.nii.gz")}
        rows.append({
            "case_id": case.name,
            "patient_id": "-".join(case.name.split("-")[:3]),  # drop the timepoint suffix
            "source": source,
            **{m: files[m].relative_to(DATA).as_posix() if m in files else "" for m in MODS},
        })
df = pd.DataFrame(rows)
df["complete"] = df[MODS].ne("").all(axis=1)

# Split GLI by PATIENT so no patient appears in two splits. Africa is held out as an external test set.
gli_patients = sorted(df.loc[df.source == "GLI", "patient_id"].unique())
train_p, rest = train_test_split(gli_patients, test_size=0.30, random_state=42)
val_p, test_p = train_test_split(rest, test_size=0.50, random_state=42)
split_of = {**{p: "train" for p in train_p}, **{p: "val" for p in val_p}, **{p: "test" for p in test_p}}
df["split"] = df.apply(lambda r: "external_test" if r.source == "SSA" else split_of[r.patient_id], axis=1)

df.to_csv(OUT / "index.csv", index=False)
print(df.groupby(["source", "split"]).size())
print("Incomplete cases:", (~df.complete).sum())