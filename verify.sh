#!/usr/bin/env bash
# verify — one command: lint + tests (with the coverage gate) + package build + smoke.
# Mirrors CI (.github/workflows/ci.yml) so a green `verify.sh` means a green CI.
# Run from the repo root:  bash verify.sh
set -euo pipefail

echo "== ruff =="
uv run ruff check .

echo "== pytest (+ coverage gate, floor from pyproject) =="
floor=$(uv run python -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['tool']['coverage']['report']['fail_under'])")
uv run pytest --cov=crucible --cov-report=term-missing --cov-fail-under="$floor"

echo "== build (wheel + sdist) =="
uv build

echo "== smoke (import + version) =="
uv run python -c "import crucible; print('crucible', crucible.__version__, 'import OK')"

echo "verify: OK"
