PYTHON ?= .venv/bin/python

.PHONY: install train train-stream submit test frontend docs doctor up down demo-ndtp acceptance-ui acceptance-cold acceptance-ndtp acceptance-faults acceptance-warm acceptance-load
install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt
	cd frontend && npm ci
train:
	$(PYTHON) -m ml.train --data-dir dataset --output-dir artifacts
train-stream:
	$(PYTHON) -m ml.train_stream
submit:
	$(PYTHON) -m ml.submit --data-dir dataset --model-dir artifacts/models --output artifacts/submission.csv
test:
	$(PYTHON) -m pytest -q
frontend:
	cd frontend && npm run build
docs:
	$(PYTHON) -m sphinx -W -b html docs docs/_build/html
up:
	$(PYTHON) -m scripts.stack up
down:
	$(PYTHON) -m scripts.stack down
doctor:
	$(PYTHON) -m scripts.stack doctor
demo-ndtp:
	$(PYTHON) -m scripts.demo_ndtp
acceptance-ui:
	$(PYTHON) scripts/check_ui_regressions.py
acceptance-cold:
	$(PYTHON) scripts/check_cold_start.py
acceptance-ndtp:
	$(PYTHON) scripts/check_ndtp.py
acceptance-faults:
	$(PYTHON) -m scripts.check_faults
acceptance-load:
	$(PYTHON) -m scripts.acceptance_load
acceptance-warm:
	$(PYTHON) -m scripts.check_warm_history
