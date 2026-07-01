#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys


TASK_NAME = "TechnicalSignalDailyUpdate"
TASK_TIME = "20:35"
COMMAND = r'wscript.exe D:\technical_signal_system\run_technical_signal_hidden.vbs'


def main() -> int:
    subprocess.run(
        [
            "schtasks",
            "/Create",
            "/TN",
            TASK_NAME,
            "/SC",
            "DAILY",
            "/ST",
            TASK_TIME,
            "/TR",
            COMMAND,
            "/F",
        ],
        check=True,
    )
    print(f"task_name={TASK_NAME}")
    print(f"task_time={TASK_TIME}")
    print("agent_status=finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
