from __future__ import annotations
import numpy as np
from src.data_types import Adapter, GPU, Request, Instance

def generate(cfg: dict, seed: int | None = None) -> Instance:
    #Build a synthetic problem instance from config dict.
    inst_cfg = cfg["instance"]
    rng = np.random.default_rng(seed if seed is not None else inst_cfg["seed"])

    adapters = _make_adapters(inst_cfg, rng)
    gpus = _make_gpus(inst_cfg, rng)
    requests = _make_requests(inst_cfg, adapters, gpus, rng)

    _check_feasibility(inst_cfg, gpus)

    return Instance(
        adapters=adapters,
        gpus=gpus,
        requests=requests,
        n_time_slots=inst_cfg["n_time_slots"],
        alpha=cfg["objective"]["alpha"],
        beta=cfg["objective"]["beta"],
        gamma=cfg["objective"]["gamma"],
    )

def _make_adapters(cfg: dict, rng: np.random.Generator) -> list[Adapter]:
    n = cfg["n_adapters"]
    mem_min = cfg["adapter_memory_gb"]["min"]
    mem_max = cfg["adapter_memory_gb"]["max"]

    adapters = []
    for i in range(n):
        memory_gb = float(rng.uniform(mem_min, mem_max))
        swap_cost = memory_gb  # 1 unit of cost per GB — proportional to size
        adapters.append(Adapter(id=i, memory_gb=memory_gb, swap_cost=swap_cost))
    return adapters

def _make_gpus(cfg: dict, rng: np.random.Generator) -> list[GPU]:
    n = cfg["n_gpus"]
    vram_min = cfg["vram_gb"]["min"]
    vram_max = cfg["vram_gb"]["max"]
    base_gb = cfg["base_model_gb"]

    gpus = []
    for j in range(n):
        vram_gb = float(rng.uniform(vram_min, vram_max))
        # faster GPUs handle more requests per slot; range 1-5
        max_throughput = int(rng.integers(1, 6))
        gpus.append(GPU(id=j, vram_gb=vram_gb, base_model_gb=base_gb,
                        max_throughput=max_throughput))

    # Guarantee that every adapter can fit on at least one GPU.
    # If not, bump the smallest GPU's VRAM to make it work.
    return gpus

def _make_requests(cfg: dict, adapters: list[Adapter], gpus: list[GPU],
                   rng: np.random.Generator) -> list[Request]:
    n = cfg["n_requests"]
    n_slots = cfg["n_time_slots"]
    priorities = cfg["request_priorities"]
    p_weights = np.array(cfg["priority_weights"], dtype=float)
    p_weights /= p_weights.sum()  # normalize to sum to 1

    # Zipf weights: adapter 0 is most popular, drops off per zipf law.
    # We draw adapter indices via these weights.
    adapter_ids = np.arange(len(adapters))
    if cfg.get("arrival", "zipf") == "zipf":
        alpha = cfg.get("zipf_alpha", 1.2)
        # raw zipf weights: 1, 1/2^a, 1/3^a, ...
        raw = np.array([(k + 1) ** (-alpha) for k in range(len(adapters))])
        adapter_weights = raw / raw.sum()
    else:
        adapter_weights = np.ones(len(adapters)) / len(adapters)

    # GPU speed: faster GPU (higher throughput) has lower base latency.
    max_tp = max(g.max_throughput for g in gpus)
    base_latency = {g.id: 1.0 + (max_tp - g.max_throughput) * 0.2 for g in gpus}

    requests = []
    for i in range(n):
        adapter_id = int(rng.choice(adapter_ids, p=adapter_weights))
        arrival_t = int(rng.integers(0, n_slots))
        priority = int(rng.choice(priorities, p=p_weights))

        # Each GPU gets a slightly different latency (noise ±10% of base).
        latency_per_gpu = tuple(
            float(base_latency[g.id] * (1.0 + rng.uniform(-0.1, 0.1)))
            for g in gpus
        )

        requests.append(Request(
            id=i,
            adapter_id=adapter_id,
            arrival_t=arrival_t,
            priority=priority,
            latency_per_gpu=latency_per_gpu,
        ))
    return requests

def _check_feasibility(cfg: dict, gpus: list[GPU]) -> None:
    #Raise if the instance is structurally infeasible before we even solve it.
    total_capacity = sum(g.max_throughput for g in gpus) * cfg["n_time_slots"]
    if cfg["n_requests"] > total_capacity:
        raise ValueError(
            f"Instance is infeasible: {cfg['n_requests']} requests but GPU cluster "
            f"can handle at most {total_capacity} across {cfg['n_time_slots']} slots. "
            "Lower n_requests or increase n_gpus / n_time_slots in config."
        )
