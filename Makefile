PYTHON ?= python

.PHONY: run-app install test

run-app:
	@test -f .env || (echo "Missing .env (copy from .env.example)" >&2; exit 1)
	set -a && . ./.env && set +a && $(PYTHON) -m uvicorn app.web:app --host 127.0.0.1 --port 8000

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest
