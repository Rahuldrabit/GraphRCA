"""GraphRCA Evaluation Runner — batch-run all AIOpsLab tasks.

Mirrors stratus/eval/eval.py: reads eval_tasks.yaml, iterates through
each task type, and runs test_graphrca.sh for each task.

Usage:
    python eval/eval.py
"""

import platform
import subprocess

import yaml


def main():
    eval_config_path = "./eval/eval_tasks.yaml"
    arch = platform.processor()
    kind_arch = "x86" if arch == "x86_64" else "arm"

    with open(eval_config_path, "r") as file:
        eval_config = yaml.safe_load(file)

    all_tasks = []
    for task_type in ("detection", "localization", "analysis", "mitigation"):
        tasks = []
        try:
            tasks = eval_config[task_type]
        except KeyError:
            print(f"Tasks for type [{task_type}] not found. Proceeding...")
        all_tasks.append((tasks, task_type))

    for tasks, task_type in all_tasks:
        if tasks is not None:
            for task in tasks:
                print(f"[EVAL-SCRIPT] running {task_type} task: {task}")
                ret = subprocess.run(
                    [
                        "/usr/bin/env",
                        "bash",
                        "./test_graphrca.sh",
                        "-r",
                        kind_arch,
                        task,
                    ]
                )
                print(
                    "[EVAL-SCRIPT] {task_type} task {task} finished with return code {returncode}, stdout: {stdout}, stderr: {stderr}".format(
                        task_type=task_type,
                        task=task,
                        returncode=ret.returncode,
                        stdout=ret.stdout,
                        stderr=ret.stderr,
                    )
                )
        else:
            print(f"[EVAL-SCRIPT] no {task_type} tasks found")


if __name__ == "__main__":
    main()
