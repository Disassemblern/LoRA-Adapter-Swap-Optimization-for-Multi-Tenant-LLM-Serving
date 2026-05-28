from __future__ import annotations
from dataclasses import dataclass
from src.data_types import Instance


@dataclass
class Solution:
    assignments: dict[int, int]                        # request_id -> gpu_id
    adapter_schedule: dict[tuple[int, int, int], bool] # (adapter_id, gpu_id, t) -> loaded?
    swaps: dict[tuple[int, int, int], bool]            # (adapter_id, gpu_id, t) -> swapped in?
    objective: float
    mip_gap: float
    wall_time_sec: float
    status: str  # "OPTIMAL", "TIME_LIMIT", "HEURISTIC"


# serialization

def solution_to_dict(sol: Solution) -> dict:
    return {
        "assignments": {str(k): v for k, v in sol.assignments.items()},
        "adapter_schedule": {
            f"{a},{j},{t}": loaded
            for (a, j, t), loaded in sol.adapter_schedule.items()
        },
        "swaps": {
            f"{a},{j},{t}": swapped
            for (a, j, t), swapped in sol.swaps.items()
        },
        "objective": sol.objective,
        "mip_gap": sol.mip_gap,
        "wall_time_sec": sol.wall_time_sec,
        "status": sol.status,
    }


def solution_from_dict(data: dict) -> Solution:
    assignments = {int(k): v for k, v in data["assignments"].items()}
    adapter_schedule = {
        tuple(int(x) for x in k.split(",")): v
        for k, v in data["adapter_schedule"].items()
    }
    swaps = {
        tuple(int(x) for x in k.split(",")): v
        for k, v in data["swaps"].items()
    }
    return Solution(
        assignments=assignments,
        adapter_schedule=adapter_schedule,  # type: ignore[arg-type]
        swaps=swaps,                        # type: ignore[arg-type]
        objective=data["objective"],
        mip_gap=data["mip_gap"],
        wall_time_sec=data["wall_time_sec"],
        status=data["status"],
    )


# verifier

def verify(sol: Solution, inst: Instance) -> list[str]:
    """
    Re-check every constraint without using Gurobi.
    Returns a list of violation strings. Empty list = valid solution.
    """
    violations: list[str] = []
    gpu_ids = [g.id for g in inst.gpus]
    adapter_map = {a.id: a for a in inst.adapters}

    # 1. Every request must be assigned to exactly one GPU.
    for req in inst.requests:
        if req.id not in sol.assignments:
            violations.append(f"Request {req.id} has no GPU assignment.")
        elif sol.assignments[req.id] not in gpu_ids:
            violations.append(
                f"Request {req.id} assigned to unknown GPU {sol.assignments[req.id]}."
            )

    # 2. The required adapter must be loaded on the assigned GPU at arrival time.
    for req in inst.requests:
        if req.id not in sol.assignments:
            continue
        j = sol.assignments[req.id]
        if not sol.adapter_schedule.get((req.adapter_id, j, req.arrival_t), False):
            violations.append(
                f"Request {req.id}: adapter {req.adapter_id} not loaded on "
                f"GPU {j} at t={req.arrival_t}."
            )

    # 3. Memory capacity: loaded adapters must not exceed each GPU's adapter budget.
    for g in inst.gpus:
        for t in range(inst.n_time_slots):
            used = sum(
                adapter_map[a.id].memory_gb
                for a in inst.adapters
                if sol.adapter_schedule.get((a.id, g.id, t), False)
            )
            if used > g.adapter_budget_gb + 1e-6:
                violations.append(
                    f"GPU {g.id} at t={t}: {used:.2f} GB used > "
                    f"budget {g.adapter_budget_gb:.2f} GB."
                )

    # 4. Throughput: GPU cannot handle more requests per slot than its limit.
    for g in inst.gpus:
        for t in range(inst.n_time_slots):
            count = sum(
                1 for req in inst.requests
                if req.arrival_t == t and sol.assignments.get(req.id) == g.id
            )
            if count > g.max_throughput:
                violations.append(
                    f"GPU {g.id} at t={t}: {count} requests > "
                    f"max_throughput {g.max_throughput}."
                )

    return violations
