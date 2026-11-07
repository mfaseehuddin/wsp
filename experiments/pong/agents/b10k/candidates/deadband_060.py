"""Candidate: the final policy.py construction with one interpretable constant changed: deadband=0.06."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b10k_policy", Path(__file__).resolve().parent.parent / "policy.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def build():
    return _mod.make(deadband=0.06)
