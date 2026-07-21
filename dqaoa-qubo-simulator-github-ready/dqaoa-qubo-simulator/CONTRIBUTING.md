# Contributing

Thank you for considering a contribution.

1. Open an issue before making a large behavioral or API change.
2. Fork the repository and create a branch from `main`.
3. Keep changes focused and include tests for corrected or new behavior.
4. Run `pytest` before opening a pull request.
5. Do not commit generated figures, solver output, credentials, proprietary data, or licensed CPLEX binaries.
6. Explain numerical-setting changes clearly because QAOA results are shot- and seed-dependent.

## Development setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[full,dev]"
pytest
```

## Pull requests

A pull request should include:

- a concise summary;
- motivation and affected modes;
- reproduction steps;
- tests or numerical evidence;
- any compatibility implications for Qiskit, Qiskit Aer, or CPLEX.
