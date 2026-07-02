#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess


LEGACY_TASKS = [
    "TechnicalSignalDailyUpdate",
    "TechnicalSignalTradingDataDaily",
    "TechnicalSignalProcessDaily",
]

TASKS = [
    {
        "name": "TechnicalSignalCalendarMonthly",
        "schedule": ["MONTHLY", "/D", "1"],
        "time": "05:00",
        "command": r'wscript.exe D:\technical_signal_system\run_technical_signal_hidden.vbs update-calendar',
    },
    {
        "name": "TechnicalSignalMarketDataDaily",
        "schedule": ["DAILY"],
        "time": "17:20",
        "command": r'wscript.exe D:\technical_signal_system\run_technical_signal_hidden.vbs update-market-data',
    },
    {
        "name": "TechnicalSignalGlobalIndexMorning",
        "schedule": ["DAILY"],
        "time": "06:40",
        "command": r'wscript.exe D:\technical_signal_system\run_technical_signal_hidden.vbs update-global-indexes',
    },
    {
        "name": "TechnicalSignalEveningPipelineDaily",
        "schedule": ["DAILY"],
        "time": "20:20",
        "command": r'wscript.exe D:\technical_signal_system\run_technical_signal_hidden.vbs evening-pipeline',
    },
]


def delete_task(name: str) -> None:
    subprocess.run(
        ["schtasks", "/Delete", "/TN", name, "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def create_task(task: dict[str, object]) -> None:
    schedule = [str(x) for x in task["schedule"]]
    subprocess.run(
        [
            "schtasks",
            "/Create",
            "/TN",
            str(task["name"]),
            "/SC",
            *schedule,
            "/ST",
            str(task["time"]),
            "/TR",
            str(task["command"]),
            "/F",
        ],
        check=True,
    )


def main() -> int:
    for task_name in LEGACY_TASKS:
        delete_task(task_name)
    for task in TASKS:
        create_task(task)
        print(f"task_name={task['name']}")
        print(f"task_time={task['time']}")
        print(f"task_command={task['command']}")
    print("legacy_tasks_removed=" + ",".join(LEGACY_TASKS))
    print("agent_status=finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
