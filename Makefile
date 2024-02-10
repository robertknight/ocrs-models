.PHONY: qa
qa: checkformat lint typecheck

.PHONY: checkformat
checkformat:
	poetry run black --check .

.PHONY: format
format:
	poetry run black ocrs_models

.PHONY: lint
lint:
	poetry run ruff ocrs_models

.PHONY: typecheck
typecheck:
	poetry run mypy ocrs_models
