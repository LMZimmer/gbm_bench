#!/usr/bin/env bash
nohup python -u scripts/evaluate_rhuh.py -algorithm sbtc > rhuh_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_rhuh.py -algorithm gliodil > rhuh_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_rhuh.py -algorithm pinngbm > rhuh_pinngbm.txt 2>&1 &

nohup python -u scripts/evaluate_gliodil.py -algorithm sbtc > gliodil_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_gliodil.py -algorithm gliodil > gliodil_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_gliodil.py -algorithm pinngbm > gliodil_pinngbm.txt 2>&1 &

nohup python -u scripts/evaluate_lumiere.py -algorithm sbtc > lumiere_sbtc.txt 2>&1 &
nohup python -u scripts/evaluate_lumiere.py -algorithm gliodil > lumiere_gliodil.txt 2>&1 &
nohup python -u scripts/evaluate_lumiere.py -algorithm pinngbm > lumiere_pinngbm.txt 2>&1 &

nohup python -u scripts/evaluate_gliomap.py -algorithm sbtc > gliomap_sbtc.txt 2>&1 &
