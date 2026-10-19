"""
Simulated Annealing for LoRA adapter swap optimization.

Solution encoding:

A solution is a 1-D integer array of length n_requests where
solution[i] = gpu_id assigned to request i.

The adapter loading schedule (y) and swap events (s) are derived
deterministically from the assignments: for each GPU and each time slot,
we greedily load all adapters needed by requests assigned to that GPU at
that slot, evicting by memory pressure (largest-first) when the VRAM
budget would be exceeded.

Objective:

objective = alpha*latency + beta*swap_cost + gamma*priority_delay
(same objective as the Gurobi model)

Constraint handling:

- Throughput: enforced by a repair step after every move.
- Memory: handled implicitly by the greedy loading heuristic; any request
  whose adapter could not be loaded incurs a large infeasibility penalty.

Cooling schedule:

Temperature decays geometrically each iteration:
  T(k) = T_start * cooling_rate^k

A worse neighbour is accepted with probability exp(-delta / T(k)).
"""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict

import numpy as np

from src.data_types import Instance
from src.solution import Solution


_INFEASIBLE_PENALTY = 1e6


class SimulatedAnnealing:
    def __init__(self, cfg: dict) -> None:
        self._t_start = float(cfg.get("t_start", 500.0))
        self._t_min = float(cfg.get("t_min", 0.1))
        self._cooling_rate = float(cfg.get("cooling_rate", 0.995))
        self._n_iterations = int(cfg.get("n_iterations", 10_000))
        self._rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self._py_rng = random.Random(int(cfg.get("seed", 0)))

    def solve(self, inst: Instance) -> Solution:
        t0 = time.time()
        gpu_ids = [g.id for g in inst.gpus]

        current = self._rng.choice(gpu_ids, size=len(inst.requests))
        current = _repair(current, inst)
        current_cost = _evaluate(current, inst)

        best = current.copy()
        best_cost = current_cost

        temperature = self._t_start

        for _ in range(self._n_iterations):
            if temperature < self._t_min:
                break

            neighbour = _neighbour(current, inst, gpu_ids, self._rng)
            neighbour_cost = _evaluate(neighbour, inst)

            delta = neighbour_cost - current_cost
            if delta < 0 or self._py_rng.random() < math.exp(-delta / temperature):
                current = neighbour
                current_cost = neighbour_cost
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost

            temperature *= self._cooling_rate

        wall_time = time.time() - t0
        return _decode(best, inst, best_cost, wall_time)


def _neighbour(
    current: np.ndarray,
    inst: Instance,
    gpu_ids: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    neighbour = current.copy()
    idx = int(rng.integers(0, len(neighbour)))
    # pick a different GPU for this request
    choices = [g for g in gpu_ids if g != neighbour[idx]]
    if choices:
        neighbour[idx] = rng.choice(choices)
    neighbour = _repair(neighbour, inst)
    return neighbour


def _repair(chrom: np.ndarray, inst: Instance) -> np.ndarray:
    chrom = chrom.copy()
    gpu_map = {g.id: g for g in inst.gpus}
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for req, gid in zip(inst.requests, chrom):
        counts[(int(gid), req.arrival_t)] += 1

    for req_idx, req in enumerate(inst.requests):
        gid = int(chrom[req_idx])
        t = req.arrival_t
        if counts[(gid, t)] > gpu_map[gid].max_throughput:
            counts[(gid, t)] -= 1
            best_gpu = min(
                inst.gpus,
                key=lambda g: counts[(g.id, t)] - gpu_map[g.id].max_throughput,
            )
            chrom[req_idx] = best_gpu.id
            counts[(best_gpu.id, t)] += 1

    return chrom


def _evaluate(chrom: np.ndarray, inst: Instance) -> float:
    assignments = {req.id: int(chrom[i]) for i, req in enumerate(inst.requests)}
    adapter_schedule, swaps = _build_schedule(assignments, inst)

    penalty = 0.0
    for req in inst.requests:
        gid = assignments[req.id]
        if not adapter_schedule.get((req.adapter_id, gid, req.arrival_t), False):
            penalty += _INFEASIBLE_PENALTY

    latency = sum(req.latency_per_gpu[assignments[req.id]] for req in inst.requests)
    swap_cost = sum(
        a.swap_cost
        for a in inst.adapters
        for g in inst.gpus
        for t in range(inst.n_time_slots)
        if swaps.get((a.id, g.id, t), False)
    )
    priority_delay = sum(
        req.priority * req.latency_per_gpu[assignments[req.id]]
        for req in inst.requests
    )

    return (
        inst.alpha * latency
        + inst.beta * swap_cost
        + inst.gamma * priority_delay
        + penalty
    )


def _build_schedule(
    assignments: dict[int, int], inst: Instance
) -> tuple[dict[tuple[int, int, int], bool], dict[tuple[int, int, int], bool]]:
    adapter_map = {a.id: a for a in inst.adapters}

    needed: dict[tuple[int, int], set[int]] = defaultdict(set)
    for req in inst.requests:
        needed[(assignments[req.id], req.arrival_t)].add(req.adapter_id)

    adapter_schedule: dict[tuple[int, int, int], bool] = {}
    swaps: dict[tuple[int, int, int], bool] = {}

    for g in inst.gpus:
        loaded: set[int] = set()
        mem_used = 0.0
        budget = g.adapter_budget_gb

        for t in range(inst.n_time_slots):
            required = needed.get((g.id, t), set())

            for aid in required:
                if aid in loaded:
                    continue
                a = adapter_map[aid]
                while mem_used + a.memory_gb > budget and loaded:
                    victim_id = max(loaded, key=lambda x: adapter_map[x].memory_gb)
                    loaded.discard(victim_id)
                    mem_used -= adapter_map[victim_id].memory_gb
                if mem_used + a.memory_gb <= budget:
                    loaded.add(aid)
                    mem_used += a.memory_gb

            prev_loaded = (
                {a.id for a in inst.adapters if adapter_schedule[(a.id, g.id, t - 1)]}
                if t > 0
                else set()
            )

            for a in inst.adapters:
                is_loaded = a.id in loaded
                adapter_schedule[(a.id, g.id, t)] = is_loaded
                if t == 0:
                    swaps[(a.id, g.id, t)] = is_loaded
                else:
                    swaps[(a.id, g.id, t)] = is_loaded and (a.id not in prev_loaded)

    return adapter_schedule, swaps


def _decode(
    chrom: np.ndarray, inst: Instance, objective: float, wall_time: float
) -> Solution:
    assignments = {req.id: int(chrom[i]) for i, req in enumerate(inst.requests)}
    adapter_schedule, swaps = _build_schedule(assignments, inst)
    return Solution(
        assignments=assignments,
        adapter_schedule=adapter_schedule,
        swaps=swaps,
        objective=objective,
        mip_gap=0.0,
        wall_time_sec=wall_time,
        status="HEURISTIC",
    )
