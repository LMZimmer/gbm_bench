import os
import ants
import shutil
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from gbm_bench.utils.constants import *
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.prediction.predict import predict_tumor_growth


if __name__ == "__main__":
    # Example:
    # python scripts/register_flair_gliodil_postop.py
    # nohup python -u scripts/register_flair_gliodil_postop.py > tmp_register_flair_gliodil_postop.out 2>&1 &

    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil_gbm = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil_gbm.load(GLIODIL_DIR)

    # Predict on preop exams
    for patient_ind, patient in enumerate(gliodil_gbm.patients):
        print(f"Predicting {patient_ind}/{len(gliodil_gbm.patients)}...")

        patient_identifier = patient["patient_id"]
        postop_exams = gliodil_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="postop")
        if len(postop_exams) < 1:
            continue
        postop_exam_dir = postop_exams[0]["t1c"].parent / "postop"
        
        flair_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=postop_exam_dir, modality="flair")
        if not flair_file.exists():
            print(f"{flair_file} not found.")
            continue

        preop_exams = gliodil_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="preop")
        preop_exam_dir = preop_exams[0]["t1c"].parent / "preop"

        t1c_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_exam_dir, modality="t1c")

        if not t1c_file.is_file():
            print(f"{t1c_file} not found")
            continue

        if not flair_file.is_file():
            print(f"{flair_file} not found")
            continue

        out_root = Path(f"/mnt/Drive2/lucas/mara_nnunet_data/GLIODIL/{patient_identifier}")
        outfile = out_root / "postop_flair_bet_normalized.nii.gz"
        adc_org_path = out_root / "flair_from_md.nii.gz"
        adc_intermediate_path = out_root / "flair_intermediate.nii.gz"

        shutil.copy(str(flair_file), str(adc_org_path))

        adc_nib = nib.load(str(adc_org_path))
        adc_data = adc_nib.get_fdata(dtype=np.float32)

        # Save intermediate image
        hdr = adc_nib.header.copy()
        hdr.set_data_dtype(np.float32)
        adc_intermediate_nib = nib.Nifti1Image(adc_data, affine=adc_nib.affine, header=hdr)
        nib.save(adc_intermediate_nib, str(adc_intermediate_path))

        # --- Step 2: Rigid registration (ANTsPy) ---
        fixed = ants.image_read(str(t1c_file))
        moving = ants.image_read(str(adc_intermediate_path))
        #reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="antsRegistrationSyN[s,2]")

        # Save registered ADC as adc.nii.gz
        ants.image_write(reg["warpedmovout"], str(outfile))

        # --- Step 3: Cleanup ---
        os.remove(str(adc_intermediate_path))
        os.remove(str(adc_org_path))

        print(f"✅ {patient_identifier}: Registration complete.")
        
    print("Done.")
