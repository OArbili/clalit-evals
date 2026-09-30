"""The examples/ scripts run against the package as pip installs it: a wheel built here, installed into a fresh
virtual environment outside the repository, with no path to this checkout."""

import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted(p.name for p in (ROOT / "examples").glob("0*.py"))


@pytest.fixture(scope="module")
def pip_python(tmp_path_factory):
    """A python whose site-packages holds only the wheel of this checkout (and pip)."""
    work = tmp_path_factory.mktemp("pip")
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(work / "dist"), str(ROOT)],
                   check=True, capture_output=True, text=True)
    (wheel,) = (work / "dist").glob("clalit_evals-*.whl")
    venv.EnvBuilder(with_pip=True).create(work / "venv")
    python = work / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", str(wheel)],
                   check=True, capture_output=True, text=True)
    return python, work


@pytest.mark.parametrize("script", EXAMPLES)
def test_example_runs_on_the_installed_package(pip_python, script):
    python, work = pip_python
    r = subprocess.run([str(python), str(ROOT / "examples" / script), "--offline"], capture_output=True, text=True,
                       timeout=120, cwd=work, env={"PATH": ""})          # no PYTHONPATH, cwd outside the repo
    assert r.returncode == 0, f"{script} failed on the installed package:\n{r.stderr[-3000:]}"
    assert "pass" in r.stdout


def test_installed_package_is_the_wheel_not_the_checkout(pip_python):
    python, work = pip_python
    r = subprocess.run([str(python), "-c", "import evals; print(evals.__file__, evals.__version__)"],
                       capture_output=True, text=True, cwd=work, env={"PATH": ""})
    assert str(work) in r.stdout and str(ROOT) not in r.stdout and "0.2.0" in r.stdout
