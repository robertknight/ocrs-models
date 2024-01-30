qa:
	poetry run black --check .
	poetry run mypy ocrs_models
