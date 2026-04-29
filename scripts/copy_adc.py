import os
import re
import shutil

# CHANGE THESE
source_root = "/mnt/Drive2/lucas/datasets/gliomap_tum_prepared_2"   # folder containing respond_XX folders
target_root = "/mnt/Drive2/lucas/mara_nnunet_data/GLIODIL"         # folder containing respond_tum_XXX folders

for patient in os.listdir(source_root):
    patient_dir = os.path.join(source_root, patient)
    if not os.path.isdir(patient_dir):
        continue

    # match only folders of the form respond_XX or respond_XXX
    if not patient.startswith("respond_"):
        continue

    # extract numeric part of the ID
    match = re.search(r"(\d+)$", patient)
    if not match:
        print(f"⚠️ Could not extract patient ID from folder name: {patient}")
        continue
    
    pid = match.group(1).zfill(3)
    target_patient_dir = os.path.join(target_root, f"respond_tum_{pid}")

    if not os.path.isdir(target_patient_dir):
        print(f"❌ Target folder not found: {target_patient_dir}")
        continue

    # --- FIXED: adc file is inside the subfolder "0" ---
    source_adc = os.path.join(patient_dir, "0", "adc.nii.gz")
    
    if not os.path.exists(source_adc):
        print(f"❌ adc.nii.gz missing in {os.path.join(patient_dir, '0')}")
        continue

    dest_adc = os.path.join(target_patient_dir, "adc.nii.gz")
    shutil.copy2(source_adc, dest_adc)
    print(f"✔ Copied {source_adc} → {dest_adc}")

