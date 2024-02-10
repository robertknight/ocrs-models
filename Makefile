.PHONY: qa
qa:
	poetry run black --check .
	poetry run mypy ocrs_models

.PHONY: format
format:
	poetry run black ocrs_models
