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
    # python scripts/register_adc_rhuh.py
    # nohup python -u scripts/-register_adc_rhuh.py > tmp_register_adc_rhuh.out 2>&1 &

    # Read dataset
    rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
    rhuh_gbm = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
    rhuh_gbm.load(RHUH_GBM_DIR)

    # Predict on preop exams
    for patient_ind, patient in enumerate(rhuh_gbm.patients):
        print(f"Predicting {patient_ind}/{len(rhuh_gbm.patients)}...")

        patient_identifier = patient["patient_id"]
        preop_exams = rhuh_gbm.get_patient_exams(patient_id=patient_identifier, timepoint="preop")
        preop_exam_dir = preop_exams[0]["t1"].parent

        t1c_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_exam_dir, modality="t1c")
        adc_file = Path("/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/") / patient_identifier / f"0/{patient_identifier}_0_adc.nii.gz"

        if not t1c_file.is_file():
            print(f"{t1c_file} not found")
            continue

        if not adc_file.is_file():
            print(f"{adc_file} not found")
            continue

        out_root = Path(f"/mnt/Drive2/lucas/mara_nnunet_data/RHUH/{patient_identifier}")
        outfile = out_root / "adc.nii.gz"
        adc_org_path = out_root / "adc_from_md.nii.gz"
        adc_intermediate_path = out_root / "adc_intermediate.nii.gz"

        shutil.copy(str(adc_file), str(adc_org_path))

        adc_nib = nib.load(str(adc_org_path))
        adc_data = adc_nib.get_fdata(dtype=np.float32)

        num_neg = np.count_nonzero(adc_data < 0)
        if num_neg > 0:
            adc_data[adc_data < 0] = 0.0
            print(f"   → Clipped {num_neg} negative voxels to 0")

        # Save intermediate image
        hdr = adc_nib.header.copy()
        hdr.set_data_dtype(np.float32)
        adc_intermediate_nib = nib.Nifti1Image(adc_data, affine=adc_nib.affine, header=hdr)
        nib.save(adc_intermediate_nib, str(adc_intermediate_path))

        # --- Step 2: Rigid registration (ANTsPy) ---
        fixed = ants.image_read(str(t1c_file))
        moving = ants.image_read(str(adc_intermediate_path))
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")

        # Save registered ADC as adc.nii.gz
        ants.image_write(reg["warpedmovout"], str(outfile))

        # --- Step 3: Cleanup ---
        os.remove(str(adc_intermediate_path))
        os.remove(str(adc_org_path))

        print(f"✅ {patient_identifier}: Registration complete.")
        
    print("Done.")
