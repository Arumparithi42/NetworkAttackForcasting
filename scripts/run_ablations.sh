#!/usr/bin/env bash
# Ablations + extra seeds for one dataset config (world model only; baselines run in 'main').
#   bash scripts/run_ablations.sh configs/cic2018_network.yaml
set -e
CFG=${1:-configs/cic2018_network.yaml}
PY=${PY:-python}
run() { tag=$1; shift; echo "== $tag"; $PY train.py --config "$CFG" --tag "$tag" --no-baselines "$@" > /dev/null; }
run seed1 --seed 1
run seed2 --seed 2
run A1_no_dynamics_loss --set train.lambdas.dyn=0
run A2_direct_heads --set model.rollout=direct
run A3_L1 --set L=1
run A3_L5 --set L=5
run A3_L20 --set L=20
run A6_teacher_forcing_only --set train.scheduled_sampling=false
run A7_lstm --set model.cell=lstm
run residual --set model.residual=true
