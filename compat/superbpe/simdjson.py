"""Minimal Python 3.14 compatibility shim for the official SuperBPE scripts.

The official project uses ``pysimdjson`` only for ``load``/``loads`` when
reading small metadata files, but pysimdjson does not publish Python 3.14
wheels. AWS uses the supported Python 3.11 dependency; this shim is local
benchmark plumbing and does not alter tokenizer training behavior.
"""

from __future__ import annotations

from json import JSONDecodeError, dump, dumps, load, loads

__all__ = ["JSONDecodeError", "dump", "dumps", "load", "loads"]

