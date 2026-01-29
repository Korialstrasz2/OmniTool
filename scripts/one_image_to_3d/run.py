import subprocess
import sys
from pathlib import Path
from venv import EnvBuilder

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / ".venv"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
APP_FILE = BASE_DIR / "app.py"


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv() -> Path:
    if not VENV_DIR.exists():
        EnvBuilder(with_pip=True).create(VENV_DIR)
    python_path = venv_python(VENV_DIR)
    subprocess.check_call([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    return python_path


def main() -> None:
    python_path = ensure_venv()
    subprocess.check_call([str(python_path), str(APP_FILE)])


if __name__ == "__main__":
    main()
