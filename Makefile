PYTHON ?= python

.PHONY: run-app install test

run-app:
	$(PYTHON) -m uvicorn app.web:app --host 127.0.0.1 --port 8000

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest
