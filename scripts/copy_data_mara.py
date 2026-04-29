import shutil
from pathlib import Path

# ===== CONFIG =====
IS_EORTC = True  # set to False to use RTOG
# ==================

def copy_unet_files(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    plan_src_name = (
        "unet_plan_eortc.nii.gz"
        if IS_EORTC
        else "unet_plan_rtog.nii.gz"
    )

    for patient_dir in src_root.iterdir():
        if not patient_dir.is_dir():
            continue

        dst_patient_dir = dst_root / patient_dir.name
        dst_patient_dir.mkdir(parents=True, exist_ok=True)

        # --- copy plan file and rename ---
        src_plan = patient_dir / plan_src_name
        dst_plan = dst_patient_dir / "unet_plan.nii.gz"

        if src_plan.exists():
            shutil.copy2(src_plan, dst_plan)
            print(f"Copied: {src_plan} -> {dst_plan}")
        else:
            print(f"Missing: {src_plan}")

        # --- copy prediction file ---
        src_pred = patient_dir / "unet_pred.nii.gz"
        dst_pred = dst_patient_dir / "unet_pred.nii.gz"

        if src_pred.exists():
            shutil.copy2(src_pred, dst_pred)
            print(f"Copied: {src_pred} -> {dst_pred}")
        else:
            print(f"Missing: {src_pred}")


if __name__ == "__main__":
    # python scripts/copy_data_mara.py
    source_root = "/mnt/Drive2/lucas/predict_gbm_fulldata_unet"
    destination_root = "/mnt/Drive2/lucas/predict_gbm_topk_eortc_fixed_gliodil"

    copy_unet_files(source_root, destination_root)

