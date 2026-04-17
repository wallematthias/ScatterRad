from __future__ import annotations

import os


def resolve_num_workers(requested: int | None, max_tasks: int | None = None) -> int:
    """Resolve worker count from explicit value or SCATTERRAD_NP environment variable."""

    workers = int(requested or 0)
    if workers <= 0:
        env_value = os.environ.get("SCATTERRAD_NP", "").strip()
        if env_value:
            try:
                workers = int(env_value)
            except ValueError:
                workers = 0
    if workers < 0:
        workers = 0
    if max_tasks is not None:
        workers = min(workers, max_tasks)
    return workers
