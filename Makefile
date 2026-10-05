# Convenience shortcuts (macOS / Linux). On Windows without `make`, run the commands directly:
# every target is just the python command shown after it (see README.md).
#
#   make setup        create .venv and install the requirements
#   make check        verify the environment (packages + mock runs of every function)
#   make test         full test-suite
#   make game         start the Syndrome Hunter game in your browser

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help setup check check-quick check-full test test-fast game bank experiments-quick experiments \
        hardware-sim notebooks clean

help:
	@grep -E '^#   make' Makefile | sed 's/^#   //'
	@echo "other targets: check-quick check-full test-fast bank experiments-quick experiments hardware-sim notebooks clean"

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "now run:  source $(VENV)/bin/activate && python -m scripts.check_env"

check:
	$(BIN)/python -m scripts.check_env

check-quick:
	$(BIN)/python -m scripts.check_env --quick

check-full:
	$(BIN)/python -m scripts.check_env --full

test:
	$(BIN)/python -m pytest

test-fast:
	$(BIN)/python -m pytest -m "not slow"

game:
	$(BIN)/streamlit run game/app.py

bank:
	$(BIN)/python -m scripts.make_game_bank

# ~2 minute smoke run with a tiny budget (numbers are NOT meaningful) -> results_quick/
experiments-quick:
	$(BIN)/python -m scripts.run_experiments --quick

# the full study from the plan (hours on a laptop; use WORKERS=4 to parallelise training)
WORKERS ?= 2
experiments:
	$(BIN)/python -m scripts.run_experiments all --workers $(WORKERS)

# offline stand-in for the IBM hardware run (no account needed; clearly labelled SIMULATED)
hardware-sim:
	$(BIN)/python -m scripts.run_hardware simulate --d 3 --rounds 3 --shots 4096

notebooks:
	$(BIN)/python -m scripts.build_notebooks

clean:
	rm -rf .pytest_cache results_quick
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
