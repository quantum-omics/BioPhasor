# BioPhasor platform — reproducibility entrypoints
PYTHON ?= .venv/bin/python

.PHONY: help env check reproduce test lint clean \
        repro-biophasor repro-phnn repro-cell-atlas repro-ehr repro-spectral

help:
	@echo "make env             create venv + install pinned deps + editable package"
	@echo "make check           verify environment + shared data cache (runs nothing)"
	@echo "make reproduce       run EVERY manuscript's experiments"
	@echo "make repro-<name>    run one manuscript (biophasor|phnn|cell-atlas|ehr|spectral)"
	@echo "make test            run the full unified pytest suite (core+phnn+spectral)"

env:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e .

check:
	bash reproduce.sh --check

reproduce:
	bash reproduce.sh

repro-biophasor:
	bash reproduce.sh biophasor
repro-phnn:
	bash reproduce.sh phnn
repro-cell-atlas:
	bash reproduce.sh cell-atlas
repro-ehr:
	bash reproduce.sh ehr
repro-spectral:
	bash reproduce.sh spectral

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf experiments/*/results/*.tmp __pycache__ */__pycache__ **/__pycache__
