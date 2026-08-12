$ErrorActionPreference = "Stop"

python -m ruff format --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m mypy src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m bandit -q -r src app scripts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m pytest
exit $LASTEXITCODE
