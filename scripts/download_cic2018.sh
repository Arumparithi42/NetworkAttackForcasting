#!/usr/bin/env bash
# Download the CSE-CIC-IDS2018 processed flow CSVs (public S3 bucket, no account needed, ~6.4 GB).
# One-time step that needs internet; everything afterwards runs offline.
set -euo pipefail
OUT=${1:-data/raw/cic2018}
mkdir -p "$OUT"
BASE="https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
for f in Wednesday-14-02-2018 Thursday-15-02-2018 Friday-16-02-2018 Thuesday-20-02-2018 \
         Wednesday-21-02-2018 Thursday-22-02-2018 Friday-23-02-2018 Wednesday-28-02-2018 \
         Thursday-01-03-2018 Friday-02-03-2018; do
  file="${f}_TrafficForML_CICFlowMeter.csv"
  if [ -s "$OUT/$file" ]; then echo "skip $file"; continue; fi
  echo "get  $file"
  curl -fsS --retry 5 -o "$OUT/$file" "$BASE/$file" &
done
wait
ls -la "$OUT"
# Equivalent with the AWS CLI:
#   aws s3 sync --no-sign-request --region ca-central-1 \
#     "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" data/raw/cic2018/
