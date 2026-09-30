#!/usr/bin/env bash
# Full reproduction after data download + PCAP extraction (see README). Logs go to logs/.
set -e
PY=${PY:-python}
mkdir -p logs
$PY scripts/preprocess.py --config configs/cic2018_network.yaml
$PY scripts/preprocess.py --config configs/cic2018_host.yaml
$PY train.py --config configs/cic2018_host.yaml --tag main > logs/host_main.log 2>&1
$PY train.py --config configs/cic2018_network.yaml --tag main > logs/net_main.log 2>&1
PY=$PY bash scripts/run_ablations.sh configs/cic2018_network.yaml > logs/net_ablations.log 2>&1
$PY train.py --config configs/cic2018_network.yaml --protocol B --tag protocolB > logs/net_protoB.log 2>&1
PY=$PY bash scripts/run_ablations.sh configs/cic2018_host.yaml > logs/host_ablations.log 2>&1
$PY scripts/evaluate.py --config configs/cic2018_network.yaml > logs/eval_network.log 2>&1
$PY scripts/evaluate.py --config configs/cic2018_host.yaml > logs/eval_host.log 2>&1
echo ALL DONE
