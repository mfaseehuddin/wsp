"""Inspect native Stage X status; orchestration belongs to the parent agent.

This script never launches an external agent. Public checks and feedback use
the one native Codex subagent through the parent agent's collaboration tools.
"""

from pathlib import Path


def main():
    status = Path(__file__).with_name("STATUS.json")
    print(status.read_text() if status.exists() else "No Stage X run registered.")


if __name__ == "__main__":
    main()
