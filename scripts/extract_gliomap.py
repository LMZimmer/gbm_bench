import os
import json
import shutil
import pickle
import argparse
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import stats
from pathlib import Path
from gbm_bench.utils.constants import GLIODIL_DIR
from gbm_bench.utils.parsing import LongitudinalDataset
from gbm_bench.evaluation.evaluate import evaluate_tumor_model
from gbm_bench.utils.constants import PREDICTION_OUTPUT_SCHEMA, MODEL_PLAN_SCHEMA, TUMORSEG_SCHEMA


def gliodil_id_to_gliomap_id(gliodil_id):
    if gliodil_id.startswith("respond"):
        number = gliodil_id.split("_")[-1]
        if number.startswith("0"):
            number = number[1:]
        return "respond_" + number
    else:
        number = gliodil_id[-2:]
        return "tgm_" + number


def binarize_prediction(pred_dir):
    outfile = pred_dir.parent / "binarized.nii.gz"
    img = nib.load(pred_dir)
    data, affine = img.get_fdata(), img.affine
    data_bin = (data>0).astype(np.int32)
    binarized = nib.Nifti1Image(data_bin, affine)
    nib.save(binarized, outfile)
    print(f"Binarized prediction saved to {outfile}")
    return outfile


def create_gliomap_predfile(probmap, coreseg):
    img1 = nib.load(probmap)
    img2 = nib.load(coreseg)

    img2_resampled = resample_from_to(img2, img1, order=0)

    # Extract data arrays
    data1 = img1.get_fdata()
    data2 = img2_resampled.get_fdata()
    
    # Add the images
    #result_data = np.clip(data1 + data2, 0, 1)
    result_data = np.clip((data1 > 0.25).astype(data2.dtype) + data2, 0, 1)


    # Create NIfTI object with original affine from probmap
    result_img = nib.Nifti1Image(result_data, img1.affine)

    # Save output
    out_path = "/mnt/Drive2/lucas/tmp/gliomappredtmp.nii.gz"
    nib.save(result_img, out_path)
    return Path(out_path)


def create_gliomap_tumorfile(tumorseg_dir):
    core_seg_nii = nib.load(str(tumorseg_dir))
    aff, header = core_seg_nii.affine, core_seg_nii.header
    core_seg = np.rint(core_seg_nii.get_fdata()).astype(np.int32)
    core_seg[core_seg==2] = 0
    core_seg[core_seg==3] = 1
    core_seg_new = nib.Nifti1Image(core_seg, affine=aff, header=header)
    out_path = "/mnt/Drive2/lucas/tmp/gliomappredtmp.nii.gz"
    nib.save(core_seg_new, out_path)
    return out_path


if __name__ == "__main__":
    # Example:
    # python scripts/extract_gliomap.py -algorithm sbtc
    # nohup python -u scripts/extract_gliomap.py -algorithm gliodil > gliomap.txt 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-algorithm", type=str, help="Algorithm ID to evaluate.")
    args = parser.parse_args()
    
    # Read dataset
    gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
    gliodil = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
    gliodil.load(GLIODIL_DIR)

    gliomap_root = Path("/mnt/Drive2/lucas/models/gliomap/gliomap_done")
    patient_ids = [p for p in os.listdir(str(gliomap_root)) if p.startswith("respond_") or p.startswith("tgm_")]
    print(f"Found {len(patient_ids)} patient ids for gliomap: {patient_ids}")

    algo_id = args.algorithm

    all_results = []
    all_results_gliomap = []
    all_results_gliomap_bin = []
    all_ids = []
    exceptions = []
    n = 0

    for patient_ind, patient in enumerate(gliodil.patients):
        print(f"Processing {patient_ind}/{len(gliodil.patients)}...")

        patient_identifier = patient["patient_id"]
        patient_identifier_gliomap = gliodil_id_to_gliomap_id(patient_identifier)

        print(patient_identifier)
        print(patient_identifier_gliomap)
        if patient_identifier_gliomap not in patient_ids:
            print(f"{patient_identifier_gliomap} not found")
            exceptions.append((patient_identifier, patient_identifier_gliomap))
            continue

        preop_exams = gliodil.get_patient_exams(patient_id=patient_identifier, timepoint="preop")
        followup_exams = gliodil.get_patient_exams(patient_id=patient_identifier, timepoint="followup")

        preop_exam_dir = preop_exams[0]["t1c"].parent / "preop"
        prediction_dir = PREDICTION_OUTPUT_SCHEMA.format(base_dir=preop_exam_dir, algo_id=algo_id)

        pbmap_dir_gliomap = gliomap_root / patient_identifier_gliomap / "0/outputs/CAT_probabilities_reg.nii.gz"
        #tumor_dir_gliomap = gliomap_root / patient_identifier_gliomap / "0/tumor.nii.gz"
        tumor_dir_gliomap = gliomap_root / patient_identifier_gliomap / "0/tumordoesnotexist.nii.gz"

        try:
            if tumor_dir_gliomap.is_file():
                print(f"Found gliomap tumor file {tumor_dir_gliomap}")
                prediction_dir_gliomap = create_gliomap_predfile(pbmap_dir_gliomap, tumor_dir_gliomap)
            else:  # fall back to tumorseg
                tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=preop_exam_dir)
                tumor_dir_gliomap = create_gliomap_tumorfile(tumorseg_file)
                prediction_dir_gliomap = create_gliomap_predfile(pbmap_dir_gliomap, tumor_dir_gliomap)
                print(f"Falling back to {tumorseg_file}")
            prediction_dir_bin_gliomap = binarize_prediction(prediction_dir_gliomap)
            n += 1
            print(n)
        except Exception as e:
            print(f"Exception for {patient_identifier}: {e}")
            exceptions.append((patient_identifier, patient_identifier_gliomap))
            continue
        
        for followup_exam in followup_exams:
            
            followup_exam_dir = followup_exam["t1c"].parent / "followup"

            try:
                results = evaluate_tumor_model(
                        preop_dir=preop_exam_dir,
                        followup_dir=followup_exam_dir,
                        pred_file=prediction_dir,
                        model_id=algo_id
                        )
                all_results.append(results)

                results_gliomap = evaluate_tumor_model(
                        preop_dir=preop_exam_dir,
                        followup_dir=followup_exam_dir,
                        pred_file=prediction_dir_gliomap,
                        model_id="gliomap_nobin",
                        )
                all_results_gliomap.append(results_gliomap)

                results_gliomap_bin = evaluate_tumor_model(
                        preop_dir=preop_exam_dir,
                        followup_dir=followup_exam_dir,
                        pred_file=prediction_dir_bin_gliomap,
                        model_id="gliomap",
                        is_binary=True
                        )
                all_results_gliomap_bin.append(results_gliomap_bin)

                # Copy model prediction
                model_pred_savepath = MODEL_PLAN_SCHEMA.format(base_dir=str(preop_exam_dir), algo_id="gliomap").parent / "gliomap_pred.nii.gz"
                shutil.copy(pbmap_dir_gliomap, model_pred_savepath)

                print(f"standard: {results['recurrence_coverage_standard']}")
                print(f"{algo_id}: {results['recurrence_coverage_model']}")
                print(f"gliomap: {results_gliomap['recurrence_coverage_model']}")
                print(f"gliomap (bin): {results_gliomap_bin['recurrence_coverage_model']}")

                all_ids.append(patient_identifier)

            except Exception as e:
                print(f"Exception for {followup_exam_dir}: {e}")

    print(f"Exceptions for {exceptions}")

    recurrence_coverage_standard = [r["recurrence_coverage_standard"] for r in all_results]
    recurrence_coverage_standard_all = [r["recurrence_coverage_standard_all"] for r in all_results]
    recurrence_coverage_model = [r["recurrence_coverage_model"] for r in all_results]
    recurrence_coverage_model_all = [r["recurrence_coverage_model_all"] for r in all_results]
    recurrence_coverage_gliomap = [r["recurrence_coverage_model"] for r in all_results_gliomap]
    recurrence_coverage_gliomap_all = [r["recurrence_coverage_model_all"] for r in all_results_gliomap]
    recurrence_coverage_gliomap_bin = [r["recurrence_coverage_model"] for r in all_results_gliomap_bin]
    recurrence_coverage_gliomap_bin_all = [r["recurrence_coverage_model_all"] for r in all_results_gliomap_bin]
    
    print(f"Total cases: {len(recurrence_coverage_standard)} / {len(recurrence_coverage_model)} / {len(recurrence_coverage_gliomap)} / {len(recurrence_coverage_gliomap_bin)}")

    print(f"Finished evaluation.")
    print(f"Standard plan coverge: {100*np.mean(recurrence_coverage_standard):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard):.2f}")
    print(f"Standard plan coverge (all): {100*np.mean(recurrence_coverage_standard_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_standard_all):.2f}")
    print(f"{algo_id.upper()} plan coverge: {100*np.mean(recurrence_coverage_model):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model):.2f}")
    print(f"{algo_id.upper()} plan coverge (all): {100*np.mean(recurrence_coverage_model_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_model_all):.2f}")
    print(f"Gliomap plan coverge: {100*np.mean(recurrence_coverage_gliomap):.2f} \u00B1 {100*stats.sem(recurrence_coverage_gliomap):.2f}")
    print(f"Gliomap plan coverge (all): {100*np.mean(recurrence_coverage_gliomap_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_gliomap_all):.2f}")
    print(f"Gliomap binarized plan coverge: {100*np.mean(recurrence_coverage_gliomap_bin):.2f} \u00B1 {100*stats.sem(recurrence_coverage_gliomap_bin):.2f}")
    print(f"Gliomap binarized plan coverge (all): {100*np.mean(recurrence_coverage_gliomap_bin_all):.2f} \u00B1 {100*stats.sem(recurrence_coverage_gliomap_bin_all):.2f}")

    # save ids
    with open("gliomap_ids.json", "w") as f:
        json.dump(all_ids, f)
