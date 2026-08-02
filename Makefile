# BioPhasor platform — reproducibility entrypoints
PYTHON ?= .venv/bin/python

.PHONY: help env check check-suites reproduce test lint format docs clean \
        repro-biophasor repro-tumor repro-spectral-classical repro-spectral-quantum \
        repro-benchmark repro-dmrg wheel check-wheel

help:
	@echo "make env             create venv + install pinned deps + editable package"
	@echo "make check           verify environment + shared data cache (runs nothing)"
	@echo "make check-suites    dry-run every suite: plan + inputs, executes NOTHING"
	@echo "make reproduce       run EVERY suite (four manuscripts, ~28 min)"
	@echo "make repro-<name>    one suite: biophasor | tumor | spectral-classical | spectral-quantum"
	@echo "make repro-benchmark spectral-classical's repeated CV + learning curve (~19 min)"
	@echo "make repro-dmrg      recompute the DMRG scan (needs physics-tenpy)"
	@echo "make test            run the full pytest suite (171 passed)"
	@echo "make lint            ruff + black --check + mypy"
	@echo "make format          ruff --fix + black"
	@echo "make wheel           build a wheel into dist/"
	@echo "make check-wheel     build a wheel and assert the quantum data ships in it"
	@echo "make docs            build the documentation site"

env:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e ".[dev,experiments]"

check:
	bash reproduce.sh --check

reproduce:
	bash reproduce.sh

repro-biophasor:
	bash reproduce.sh biophasor
repro-tumor:
	bash reproduce.sh tumor
repro-spectral-classical:
	bash reproduce.sh spectral-classical
repro-spectral-quantum:
	bash reproduce.sh spectral-quantum

# Dry run: validates each suite's plan, inputs and output dirs, runs nothing.
check-suites:
	PYTHONPATH=. $(PYTHON) -m experiments.biophasor.codes.run_all --check
	PYTHONPATH=. $(PYTHON) -m experiments.tumor.codes.run_all --check
	PYTHONPATH=. $(PYTHON) experiments/spectral-classical/codes/run_all.py --check
	PYTHONPATH=. $(PYTHON) experiments/spectral-quantum/codes/run_all.py --check

# Slow, flag-gated steps, deliberately off the default reproduce path.
repro-benchmark:
	PYTHONPATH=. $(PYTHON) experiments/spectral-classical/codes/run_all.py --with-benchmark
repro-dmrg:
	PYTHONPATH=. $(PYTHON) experiments/spectral-quantum/codes/run_all.py --dmrg

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check biophasor experiments tests
	$(PYTHON) -m black --check biophasor tests
	$(PYTHON) -m mypy biophasor

format:
	$(PYTHON) -m black biophasor tests
	$(PYTHON) -m ruff check --fix biophasor experiments tests

docs:
	$(PYTHON) -m mkdocs build --strict

wheel:
	rm -rf dist
	$(PYTHON) -m pip wheel . --no-deps -w dist

# The omics harmonic ladder is package data, and its absence does not raise:
# compartment_self_energies() falls through to a synthetic spectrum and returns
# a plausible but wrong epsilon ladder. An editable install cannot show this,
# because the file is still on disk beside the source — so the only way to see
# the defect is to look inside a real wheel. Run this before any release.
check-wheel: wheel
	@$(PYTHON) -c "import glob, sys, zipfile; \
	w = glob.glob('dist/*.whl')[0]; \
	n = zipfile.ZipFile(w).namelist(); \
	want = ['biophasor/spectral/quantum/data/omega_k.npy', \
	        'biophasor/spectral/quantum/data/omega_provenance.json']; \
	miss = [f for f in want if f not in n]; \
	sys.exit('MISSING from %s: %s\n  check [tool.setuptools.package-data]' % (w, miss)) if miss else \
	print('%s: quantum package data present' % w)"

clean:
	rm -rf experiments/*/results/*.tmp dist site .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
