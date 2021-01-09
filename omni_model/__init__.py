from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
REPO_ROOT = PACKAGE_ROOT.parent

SRC = PACKAGE_ROOT / "src"
WEIGHTS = PACKAGE_ROOT / "weights"
TESTS = PACKAGE_ROOT / "tests"
