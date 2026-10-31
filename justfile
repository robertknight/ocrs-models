python_srcs := "ocrs_models scripts"

qa: checkformat lint typecheck

checkformat:
    uv run ruff format --check {{python_srcs}}

format:
    uv run ruff format {{python_srcs}}

lint:
    uv run ruff check {{python_srcs}}

typecheck:
    uv run mypy {{python_srcs}}

# Train the text detection model on a remote GPU host via SSH.
# Usage: just train-detection-ssh user@host [training args...]
train-detection-ssh host *args:
    uv run --no-project python scripts/train_remote.py detection {{host}} {{args}}

# Train the text recognition model on a remote GPU host via SSH.
# Usage: just train-recognition-ssh user@host [training args...]
train-recognition-ssh host *args:
    uv run --no-project python scripts/train_remote.py recognition {{host}} {{args}}
