"""
A copy of the console in a file: everything written to stdout and stderr (the progress lines,
the evaluation tables, warnings and tracebacks) also goes to the file, so a run started from an
IDE or without ``tee`` keeps its console log next to its checkpoints.
"""

import sys
from datetime import datetime
from pathlib import Path

__all__ = ["tee_console"]


class _Tee:
    """A text stream that writes to a terminal stream and a file, flushing the file per write."""

    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno, ...: the terminal's
        return getattr(self.stream, name)


def tee_console(path) -> None:
    """
    From now on, copy stdout and stderr to ``path`` (appended to, with a dated header, so a
    resumed run continues the same file). Call once, on the process that prints.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("a", encoding="utf-8", buffering=1)
    file.write(f"\n===== {datetime.now().isoformat(timespec='seconds')}  {' '.join(sys.argv)}\n")
    sys.stdout = _Tee(sys.stdout, file)
    sys.stderr = _Tee(sys.stderr, file)
