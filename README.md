# clalit-evals

A minimal Python package with a single function, `hello_world()`, that prints `Hello, World!`.

## Installation

```bash
pip install clalit-evals
```

## Usage

```python
from clalit_evals import hello_world

hello_world()
# Hello, World!
```

## Development

Create a virtual environment, install the package in editable mode with the dev tools, and run the tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Building and publishing

Build the source distribution and wheel, then check the metadata:

```bash
python -m build
twine check dist/*
```

Upload to TestPyPI first (recommended), then to PyPI. Both need an API token from the
respective site, used as the password with the username `__token__`:

```bash
twine upload --repository testpypi dist/*
twine upload dist/*
```

Alternatively, `.github/workflows/publish.yml` publishes automatically when a GitHub release
is published. It uses PyPI trusted publishing, so no token is stored in GitHub: on PyPI, go to
*Your account → Publishing* and add this repository with workflow name `publish.yml` and
environment name `pypi`.

Bump `version` in `pyproject.toml` and add an entry to `CHANGELOG.md` before each release.

## License

[MIT](LICENSE)
