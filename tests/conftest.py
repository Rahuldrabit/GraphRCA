import sys
from pathlib import Path


def pytest_configure() -> None:
    pkg_dir = Path(__file__).resolve().parents[1]  # .../GraphRCA_agent
    repo_dir = pkg_dir.parent                      # .../GraphRCA

    repo_dir_str = str(repo_dir)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)