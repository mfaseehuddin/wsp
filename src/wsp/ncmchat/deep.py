"""wsp-ncmdeep: the deep NCM chat round. Same harness, experiments/ncmdeep, K >= 4 levels with stated meanings."""

from . import harness


def main(argv=None):
    harness.ROOT = harness.REPO / "experiments/ncmdeep"
    harness.MIN_LEVELS = 4
    harness.main(argv)
