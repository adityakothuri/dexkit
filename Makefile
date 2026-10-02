# DexKit build targets. Works on Ubuntu 22.04 and macOS.
PY ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
VENV := .venv
BIN := $(VENV)/bin
UNAME := $(shell uname -s)

.PHONY: setup setup-core test lint ports calibrate demo mock-all policy-smoke clean

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[policy,dev]"
ifeq ($(UNAME),Linux)
	sudo cp scripts/99-dexkit.rules /etc/udev/rules.d/99-dexkit.rules
	sudo udevadm control --reload-rules && sudo udevadm trigger
	@echo "add yourself to dialout if needed: sudo usermod -aG dialout $$USER (then log out/in)"
else
	@echo "udev rule skipped on $(UNAME) (Linux only)"
endif

setup-core:  ## no torch: hardware layer + tests only
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests

# Which USB port is the hand and which is the gantry (plug both in first).
ports:
	@$(BIN)/python -c "from dexkit.hw.ports import *; ps = list_usb_ports(); print(describe_ports(ps)); \
	print('hand   ->', find_port('hand', ps) or 'NOT FOUND'); print('gantry ->', find_port('gantry', ps) or 'NOT FOUND')"

calibrate:
	$(BIN)/dexkit-scan
	$(BIN)/dexkit-calibrate-hand
	$(BIN)/dexkit-calibrate-gantry

demo:
	$(BIN)/dexkit-run examples/pick_and_show.yaml

# Every CLI end to end on simulated hardware (writes to a temp data dir).
mock-all:
	DEXKIT_DATA=$$(mktemp -d) sh -c '\
	  set -e; \
	  $(BIN)/dexkit-scan --mock; \
	  $(BIN)/dexkit-calibrate-hand --mock --scripted --force; \
	  $(BIN)/dexkit-calibrate-gantry --mock --scripted; \
	  $(BIN)/dexkit-pose --mock --yes fist; \
	  $(BIN)/dexkit-teleop --mock --yes --duration 10 --script "0.5:3,1:],1.5:.,2:w,3:r,3.5:f,5:o,6:r,9:esc"; \
	  $(BIN)/dexkit-replay --mock --yes $$(ls -t $$DEXKIT_DATA/recordings | head -1); \
	  $(BIN)/dexkit-run --mock examples/pick_and_show.yaml --dry-run; \
	  $(BIN)/dexkit-run --mock --yes --mock-speed 5 examples/pick_and_show.yaml; \
	  $(BIN)/dexkit-run --mock --yes --loop 2 examples/finger_ripple.yaml; \
	  $(BIN)/dexkit-run --mock --yes examples/show_off.yaml; \
	  $(BIN)/dexkit-relax --mock --yes; \
	  $(BIN)/dexkit-estop --mock; \
	  echo "mock-all: every CLI OK"'

policy-smoke:
	$(BIN)/python -m dexkit.policy.smoke

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info src/*.egg-info
