import os
import re
import shutil

# CHANGE THESE
source_root = "/mnt/Drive2/lucas/datasets/GLIODIL"  # contains folders: tgm001, tgm002, ...
target_root = "/mnt/Drive2/lucas/mara_nnunet_data/GLIODIL" # contains folders: tgm001, tgm002, ...

for patient in os.listdir(source_root):
    source_patient_dir = os.path.join(source_root, patient)
    if not os.path.isdir(source_patient_dir):
        continue

    # expect folders like: tgm001
    match = re.match(r"(tgm\d+)$", patient)
    if not match:
        continue
    pid = match.group(1)  # e.g. tgm001

    target_patient_dir = os.path.join(target_root, pid)
    if not os.path.isdir(target_patient_dir):
        print(f"❌ Target folder missing: {target_patient_dir}")
        continue

    # FET PET is located in a subfolder "preop"
    preop_dir = os.path.join(source_patient_dir, "preop")
    if not os.path.isdir(preop_dir):
        print(f"❌ Missing preop folder for {patient}")
        continue

    # file is "sub-tgm001_ses-preop_space-sri_fet.nii.gz"
    fet_filename = f"sub-{pid}_ses-preop_space-sri_fet.nii.gz"
    source_fet = os.path.join(preop_dir, fet_filename)

    if not os.path.exists(source_fet):
        print(f"❌ FET PET file missing: {source_fet}")
        continue

    dest_fet = os.path.join(target_patient_dir, "fet.nii.gz")  # or keep same filename if preferred
    shutil.copy2(source_fet, dest_fet)
    print(f"✔ Copied {source_fet} → {dest_fet}")

