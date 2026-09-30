PY ?= python
NET = configs/cic2018_network.yaml
HOST = configs/cic2018_host.yaml

.PHONY: install data pcap preprocess train ablations evaluate app test demo

install:
	$(PY) -m pip install -r requirements.txt

data:                ## download the processed CSVs (~6.4 GB, one-time, internet)
	bash scripts/download_cic2018.sh

pcap:                ## stream + extract the two infiltration days' PCAPs (~116 GB streamed, ~1 GB kept)
	$(PY) scripts/extract_cic2018_pcaps.py --day Thursday-01-03-2018
	$(PY) scripts/extract_cic2018_pcaps.py --day Wednesday-28-02-2018

preprocess:
	$(PY) scripts/preprocess.py --config $(NET)
	$(PY) scripts/preprocess.py --config $(HOST)

train:
	$(PY) train.py --config $(NET)
	$(PY) train.py --config $(HOST)

ablations:
	PY=$(PY) bash scripts/run_ablations.sh $(NET)
	$(PY) train.py --config $(NET) --protocol B --tag protocolB

evaluate:
	$(PY) scripts/evaluate.py --config $(NET)
	$(PY) scripts/evaluate.py --config $(HOST)

demo:                ## refresh the committed demo/ bundle from local artifacts
	$(PY) scripts/make_demo_bundle.py

app:
	streamlit run app.py

test:
	$(PY) -m pytest -q tests
