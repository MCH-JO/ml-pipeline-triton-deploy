.PHONY: venv test infer

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

# Paramètres Triton (tu peux les overrider)
TRITON_URL ?= localhost:8000
TRITON_MODEL ?= simple
TRITON_INPUT ?= input
TRITON_OUTPUT ?= output

venv:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r client/requirements.txt

test:
	TRITON_URL=$(TRITON_URL) \
	TRITON_MODEL=$(TRITON_MODEL) \
	TRITON_INPUT=$(TRITON_INPUT) \
	TRITON_OUTPUT=$(TRITON_OUTPUT) \
	$(PY) -m pytest -q client/tests

infer:
	$(PY) client/infer.py --url $(TRITON_URL) --model $(TRITON_MODEL) --input-name $(TRITON_INPUT) --output-name $(TRITON_OUTPUT)
