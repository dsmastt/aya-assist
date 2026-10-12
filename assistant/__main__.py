"""Allow ``python -m assistant`` to launch the CLI."""

from __future__ import annotations

import sys

from assistant.cli import main

if __name__ == "__main__":
    sys.exit(main())
