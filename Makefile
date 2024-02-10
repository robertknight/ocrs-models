.PHONY: qa
qa: checkformat lint typecheck

PYTHON_SRCS=ocrs_models

.PHONY: checkformat
checkformat:
	poetry run ruff format --check $(PYTHON_SRCS)

.PHONY: format
format:
	poetry run ruff format $(PYTHON_SRCS)

.PHONY: lint
lint:
	poetry run ruff $(PYTHON_SRCS)

.PHONY: typecheck
typecheck:
	poetry run mypy $(PYTHON_SRCS)
