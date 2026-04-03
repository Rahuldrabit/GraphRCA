"""Strip ANSI escape sequences from log files.

Usage:
    python eval/clean_ansi_from_log.py <logfile>

Mirrors stratus/eval/clean_asci_color_from_log.py.
"""

import re
import sys


def clean_ansi(filepath: str):
    ansi_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    with open(filepath, "r") as f:
        text = f.read()
    with open(filepath, "w") as f:
        f.write(ansi_re.sub("", text))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile>")
        sys.exit(1)
    clean_ansi(sys.argv[1])
