python_srcs := "ocrs_models"

qa: checkformat lint typecheck

checkformat:
    uv run ruff format --check {{python_srcs}}

format:
    uv run ruff format {{python_srcs}}

lint:
    uv run ruff check {{python_srcs}}

typecheck:
    uv run mypy {{python_srcs}}
