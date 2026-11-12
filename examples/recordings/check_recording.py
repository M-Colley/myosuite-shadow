"""Audit/reprocess external recordings without bundling participant data.

Run with the installed project environment, for example:
python examples/recordings/check_recording.py audit /data/frozen.json
"""

from myosuite_shadow.recordings_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
