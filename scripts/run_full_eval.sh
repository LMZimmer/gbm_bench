#!/usr/bin/env bash
nohup python -u scripts/evaluate_rhuh.py -algorithm sbtc > rhuh_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_rhuh.py -algorithm gliodil > rhuh_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_rhuh.py -algorithm lmi > rhuh_lmi.txt 2>&1 &

nohup python -u scripts/evaluate_upenngbm.py -algorithm sbtc > upenn_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_upenngbm.py -algorithm gliodil > upenn_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_upenngbm.py -algorithm lmi > upenn_lmi.txt 2>&1 &

nohup python -u scripts/evaluate_gliodil.py -algorithm sbtc > gliodil_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_gliodil.py -algorithm gliodil > gliodil_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_gliodil.py -algorithm lmi > gliodil_lmi.txt 2>&1 &

nohup python -u scripts/evaluate_lumiere.py -algorithm sbtc > lumiere_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_lumiere.py -algorithm gliodil > lumiere_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_lumiere.py -algorithm lmi > lumiere_lmi.txt 2>&1 &

nohup python -u scripts/evaluate_ivygap.py -algorithm sbtc > ivygap_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_ivygap.py -algorithm gliodil > ivygap_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_ivygap.py -algorithm lmi > ivygap_lmi.txt 2>&1 &

nohup python -u scripts/evaluate_cptac.py -algorithm sbtc > cptac_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_cptac.py -algorithm gliodil > cptac_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_cptac.py -algorithm lmi > cptac_lmi.txt 2>&1 &

nohup python -u scripts/evaluate_tcga_gbm.py -algorithm sbtc > tcga_gbm_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_tcga_gbm.py -algorithm gliodil > tcga_gbm_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_tcga_gbm.py -algorithm lmi > tcga_gbm_lmi.txt 2>&1 &

nohup python -u scripts/evaluate_tcga_lgg.py -algorithm sbtc > tcga_lgg_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_tcga_lgg.py -algorithm gliodil > tcga_lgg_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_tcga_lgg.py -algorithm lmi > tcga_lgg_lmi.txt 2>&1 &

nohup python -u scripts/evaluate_nnUnet.py -dataset rhuh > rhuh_nnUnet.txt 2>&1 &
nohup python -u scripts/evaluate_nnUnet.py -dataset upenn > upenn_nnUnet.txt 2>&1 &
nohup python -u scripts/evaluate_nnUnet.py -dataset lumiere > lumiere_nnUnet.txt 2>&1 &
nohup python -u scripts/evaluate_nnUnet.py -dataset gliodil > gliodil_nnUnet.txt 2>&1 &
nohup python -u scripts/evaluate_nnUnet.py -dataset ivygap > ivygap_nnUnet.txt 2>&1 &
nohup python -u scripts/evaluate_nnUnet.py -dataset cptac > cptac_nnUnet.txt 2>&1 &
nohup python -u scripts/evaluate_nnUnet.py -dataset tcga-gbm > tcga_gbm_nnUnet.txt 2>&1 &
nohup python -u scripts/evaluate_nnUnet.py -dataset tcga-lgg > tcga_lgg_nnUnet.txt 2>&1 &

nohup python scripts/evaluate_datasets.py -algorithm sbtc > full_sbtc.txt 2>&1 &
nohup python scripts/evaluate_datasets.py -algorithm gliodil > full_gliodil.txt 2>&1 &
nohup python scripts/evaluate_datasets.py -algorithm lmi > full_lmi.txt 2>&1 &
nohup python scripts/evaluate_datasets.py -algorithm nnUnet > full_nnunet.txt 2>&1 &
