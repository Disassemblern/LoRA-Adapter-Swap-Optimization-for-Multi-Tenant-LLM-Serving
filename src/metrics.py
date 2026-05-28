from __future__ import annotations
from src.data_types import Instance
from src.solution import Solution


def decompose(sol: Solution, inst: Instance) -> dict[str, float]:
    """
    Break the total cost into three named components.
    Returns {"latency": ..., "swap": ..., "priority": ..., "total": ...}

    We recompute from the solution dicts rather than trusting sol.objective
    because heuristic solutions compute their own objective approximation.
    """
    latency = sum(
        req.latency_per_gpu[sol.assignments[req.id]]
        for req in inst.requests
        if req.id in sol.assignments
    )

    swap = sum(
        a.swap_cost
        for a in inst.adapters
        for g in inst.gpus
        for t in range(inst.n_time_slots)
        if sol.swaps.get((a.id, g.id, t), False)
    )

    priority = sum(
        req.priority * req.latency_per_gpu[sol.assignments[req.id]]
        for req in inst.requests
        if req.id in sol.assignments
    )

    total = inst.alpha * latency + inst.beta * swap + inst.gamma * priority

    return {
        "latency": round(latency, 4),
        "swap": round(swap, 4),
        "priority": round(priority, 4),
        "total": round(total, 4),
    }
