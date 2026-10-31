"""
Train a model on a remote host with an NVIDIA GPU via SSH.

Code and the HierText dataset are copied to the remote host using rsync, the
model is trained and exported to ONNX there, and the checkpoint and ONNX model
are then copied back to the root of this repository.

The remote host needs passwordless SSH login, rsync and uv. If `WANDB_API_KEY`
is set locally, it is forwarded to the remote host.
"""

import os
import shlex
import subprocess
from argparse import REMAINDER, ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILES = [
    "datasets/hiertext/gt/train.jsonl.gz",
    "datasets/hiertext/gt/validation.jsonl.gz",
    "datasets/hiertext/train",
    "datasets/hiertext/validation",
]

MODELS = {
    "detection": {
        "module": "ocrs_models.train_detection",
        "checkpoint": "text-detection-checkpoint.pt",
        "onnx": "text-detection.onnx",
        "default_args": ["--max-epochs", "50", "--batch-size", "28"],
    },
    "recognition": {
        "module": "ocrs_models.train_rec",
        "checkpoint": "text-rec-checkpoint.pt",
        "onnx": "text-recognition.onnx",
        "default_args": ["--max-epochs", "50", "--batch-size", "250"],
    },
}


def run(cmd: list[str], input: str | None = None):
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, input=input, text=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS.keys())
    parser.add_argument("host", help="SSH destination, eg. user@host")
    parser.add_argument(
        "--remote-dir",
        default="ocrs-models",
        help="Directory on the remote host, relative to the remote home directory",
    )
    parser.add_argument(
        "train_args",
        nargs=REMAINDER,
        help="Extra arguments for the training script, which override the defaults",
    )
    args = parser.parse_args()

    model = MODELS[args.model]
    host = args.host
    remote_dir = args.remote_dir

    for path in DATASET_FILES:
        if not (REPO_ROOT / path).exists():
            raise SystemExit(f"Dataset file {path} not found. See docs/training.md.")

    # Note: macOS ships openrsync, which supports only a subset of rsync's
    # options, so stick to basic flags here.
    print(f"Copying code to {host}:{remote_dir}")
    code_files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    run(["ssh", host, "mkdir", "-p", remote_dir])
    run(["rsync", "-a", "--relative", *code_files, f"{host}:{remote_dir}/"])

    print(f"Copying dataset to {host}:{remote_dir}. This is slow the first time.")
    run(["rsync", "-a", "--relative", *DATASET_FILES, f"{host}:{remote_dir}/"])

    # Run the remote commands with bash from an uploaded script, so they don't
    # depend on the remote user's login shell or need another level of quoting
    # for SSH.
    train_cmd = [
        "uv",
        "run",
        "python",
        "-m",
        model["module"],
        "hiertext",
        "datasets/hiertext/",
    ]
    train_args = model["default_args"] + args.train_args
    export_args = ["--checkpoint", model["checkpoint"], "--export", model["onnx"]]
    env_lines = []
    if wandb_key := os.environ.get("WANDB_API_KEY"):
        env_lines.append(f"export WANDB_API_KEY={shlex.quote(wandb_key)}")
    remote_script = "\n".join(
        [
            "set -euo pipefail",
            *env_lines,
            f"cd {shlex.quote(remote_dir)}",
            'export PATH="$HOME/.local/bin:$PATH"',
            "uv sync --locked",
            shlex.join(train_cmd + train_args),
            shlex.join(train_cmd + export_args),
        ]
    )
    script_path = shlex.quote(f"{remote_dir}/.train-remote.sh")
    run(["ssh", host, f"umask 077; cat > {script_path}"], input=remote_script)

    # Allocate a terminal so that progress bars can determine the terminal
    # width and update in place, and so that Ctrl+C stops the remote process.
    print(f"Training {args.model} model on {host}")
    run(["ssh", "-t", host, "bash", script_path])

    print(f"Copying {model['checkpoint']} and {model['onnx']} from {host}")
    run(
        [
            "rsync",
            "-a",
            f"{host}:{remote_dir}/{model['checkpoint']}",
            f"{host}:{remote_dir}/{model['onnx']}",
            ".",
        ]
    )


if __name__ == "__main__":
    main()
