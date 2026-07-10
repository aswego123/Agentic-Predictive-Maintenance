"""
Ensure the parent directory (which holds the legacy hyphenated files)
is on sys.path so `_legacy_imports` works when pytest is invoked from
inside the package directory.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_PKG_ROOT)
for p in (_REPO_ROOT, os.path.dirname(_REPO_ROOT)):
    if p and p not in sys.path:
        sys.path.insert(0, p)
