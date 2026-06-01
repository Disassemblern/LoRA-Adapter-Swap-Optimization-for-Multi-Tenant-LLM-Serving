"""
Genetic Algorithm for LoRA adapter swap optimization.

Chromosome encoding:

A chromosome is a 1-D integer array of length n_requests where
chromosome[i] = gpu_id assigned to request i.

The adapter loading schedule (y) and swap events (s) are derived
deterministically from the assignments: for each GPU and each time slot,
we greedily load all adapters needed by requests assigned to that GPU at
that slot, evicting by memory pressure (largest-first) when the VRAM
budget would be exceeded.

Fitness:

fitness = alpha*latency + beta*swap_cost + gamma*priority_delay
(same objective as the Gurobi model)

Constraint handling:

Hard constraints are enforced via repair:
  - Throughput: if a GPU has too many requests at time t, excess requests
    are re-assigned to the GPU with the most remaining capacity at that slot.
  - Memory: handled implicitly by the greedy loading heuristic (adapters
    that don't fit are simply not loaded, making the request infeasible).
    The fitness function adds a large penalty for any unserviced request.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Sequence

import numpy as np

from src.data_types import Instance, Adapter, GPU, Request
from src.solution import Solution


_INFEASIBLE_PENALTY = 1e6


class GeneticAlgorithm:
    def __init__(self, cfg: dict) -> None:
        self._pop_size = int(cfg.get("population_size", 60))
        self._n_gen = int(cfg.get("n_generations", 200))
        self._cx_rate = float(cfg.get("crossover_rate", 0.85))
        self._mut_rate = float(cfg.get("mutation_rate", 0.02))
        self._tourn_size = int(cfg.get("tournament_size", 4))
        self._elite = int(cfg.get("elite_count", 2))
        self._rng = np.random.default_rng(int(cfg.get("seed", 0)))

    def solve(self, inst: Instance) -> Solution:
        t0 = time.time()
        gpu_ids = [g.id for g in inst.gpus]
        n_req = len(inst.requests)

        pop = self._init_population(inst, gpu_ids, n_req)
        fits = np.array([self._fitness(chrom, inst) for chrom in pop])

        best_chrom = pop[int(np.argmin(fits))].copy()
        best_fit = float(np.min(fits))

        for _ in range(self._n_gen):
            # elitism: carry top individuals unchanged
            elite_idx = np.argsort(fits)[: self._elite]
            new_pop: list[np.ndarray] = [pop[i].copy() for i in elite_idx]

            while len(new_pop) < self._pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1, gpu_ids)
                c2 = self._mutate(c2, gpu_ids)
                c1 = self._repair(c1, inst)
                c2 = self._repair(c2, inst)
                new_pop.append(c1)
                if len(new_pop) < self._pop_size:
                    new_pop.append(c2)

            pop = new_pop
            fits = np.array([self._fitness(chrom, inst) for chrom in pop])

            gen_best = float(np.min(fits))
            if gen_best < best_fit:
                best_fit = gen_best
                best_chrom = pop[int(np.argmin(fits))].copy()

        wall_time = time.time() - t0
        return _decode(best_chrom, inst, best_fit, wall_time)

    # population initialisation

    def _init_population(
        self, inst: Instance, gpu_ids: list[int], n_req: int
    ) -> list[np.ndarray]:
        pop = []
        for _ in range(self._pop_size):
            chrom = self._rng.choice(gpu_ids, size=n_req)
            chrom = self._repair(chrom, inst)
            pop.append(chrom)
        return pop

    # selection

    def _tournament(self, pop: list[np.ndarray], fits: np.ndarray) -> np.ndarray:
        candidates = self._rng.choice(len(pop), size=self._tourn_size, replace=False)
        winner = candidates[int(np.argmin(fits[candidates]))]
        return pop[winner]

    # crossover (uniform)

    def _crossover(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._rng.random() > self._cx_rate:
            return p1.copy(), p2.copy()
        mask = self._rng.random(len(p1)) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    # mutation

    def _mutate(self, chrom: np.ndarray, gpu_ids: list[int]) -> np.ndarray:
        chrom = chrom.copy()
        for i in range(len(chrom)):
            if self._rng.random() < self._mut_rate:
                chrom[i] = self._rng.choice(gpu_ids)
        return chrom

    # repair: fix throughput violations

    def _repair(self, chrom: np.ndarray, inst: Instance) -> np.ndarray:
        chrom = chrom.copy()
        gpu_map = {g.id: g for g in inst.gpus}

        # count requests per (gpu, t)
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for req, gid in zip(inst.requests, chrom):
            counts[(int(gid), req.arrival_t)] += 1

        for req_idx, req in enumerate(inst.requests):
            gid = int(chrom[req_idx])
            t = req.arrival_t
            if counts[(gid, t)] > gpu_map[gid].max_throughput:
                counts[(gid, t)] -= 1
                # find gpu with most remaining capacity at this slot
                best_gpu = min(
                    inst.gpus,
                    key=lambda g: counts[(g.id, t)] - gpu_map[g.id].max_throughput,
                )
                chrom[req_idx] = best_gpu.id
                counts[(best_gpu.id, t)] += 1

        return chrom

    # fitness

    def _fitness(self, chrom: np.ndarray, inst: Instance) -> float:
        assignments = {req.id: int(chrom[i]) for i, req in enumerate(inst.requests)}
        adapter_schedule, swaps = _build_schedule(assignments, inst)

        penalty = 0.0
        for req in inst.requests:
            gid = assignments[req.id]
            if not adapter_schedule.get((req.adapter_id, gid, req.arrival_t), False):
                penalty += _INFEASIBLE_PENALTY

        latency = sum(
            req.latency_per_gpu[assignments[req.id]] for req in inst.requests
        )
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


# schedule builder (deterministic, shared by fitness and decode)

def _build_schedule(
    assignments: dict[int, int], inst: Instance
) -> tuple[dict[tuple[int, int, int], bool], dict[tuple[int, int, int], bool]]:
    """
    For each GPU, replay time slots in order. At each slot load all adapters
    needed by requests assigned to that GPU at that slot. Evict the largest
    adapter when memory is full (greedy, not LRU — keeps the schedule simple
    and swap-count low).
    """
    adapter_map = {a.id: a for a in inst.adapters}

    # requests needed per (gpu, t)
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
                # evict largest adapter until the new one fits
                while mem_used + a.memory_gb > budget and loaded:
                    victim_id = max(loaded, key=lambda x: adapter_map[x].memory_gb)
                    loaded.discard(victim_id)
                    mem_used -= adapter_map[victim_id].memory_gb
                if mem_used + a.memory_gb <= budget:
                    loaded.add(aid)
                    mem_used += a.memory_gb

            prev_loaded = {
                aid for aid in inst.adapters
                if adapter_schedule.get((aid, g.id, t - 1), False)
            } if t > 0 else set()

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
