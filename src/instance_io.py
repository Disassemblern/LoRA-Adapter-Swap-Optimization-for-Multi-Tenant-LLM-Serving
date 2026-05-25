from __future__ import annotations
import json
from pathlib import Path
from src.data_types import Adapter, GPU, Request, Instance

SCHEMA_VERSION = 1

def save(instance: Instance, path: str | Path) -> None:
    #Save an Instance to a JSON file.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": SCHEMA_VERSION,
        "n_time_slots": instance.n_time_slots,
        "alpha": instance.alpha,
        "beta": instance.beta,
        "gamma": instance.gamma,
        "adapters": [
            {"id": a.id, "memory_gb": a.memory_gb, "swap_cost": a.swap_cost}
            for a in instance.adapters
        ],
        "gpus": [
            {"id": g.id, "vram_gb": g.vram_gb,
             "base_model_gb": g.base_model_gb, "max_throughput": g.max_throughput}
            for g in instance.gpus
        ],
        "requests": [
            {"id": r.id, "adapter_id": r.adapter_id, "arrival_t": r.arrival_t,
             "priority": r.priority, "latency_per_gpu": list(r.latency_per_gpu)}
            for r in instance.requests
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load(path: str | Path) -> Instance:
    #Loads an Instance from a JSON file.
    with open(path) as f:
        data = json.load(f)

    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version: {data.get('schema_version')}. "
            f"Expected {SCHEMA_VERSION}."
        )

    adapters = [
        Adapter(id=a["id"], memory_gb=a["memory_gb"], swap_cost=a["swap_cost"])
        for a in data["adapters"]
    ]
    gpus = [
        GPU(id=g["id"], vram_gb=g["vram_gb"],
            base_model_gb=g["base_model_gb"], max_throughput=g["max_throughput"])
        for g in data["gpus"]
    ]
    requests = [
        Request(
            id=r["id"],
            adapter_id=r["adapter_id"],
            arrival_t=r["arrival_t"],
            priority=r["priority"],
            latency_per_gpu=tuple(r["latency_per_gpu"]),  # list -> tuple on load
        )
        for r in data["requests"]
    ]

    return Instance(
        adapters=adapters,
        gpus=gpus,
        requests=requests,
        n_time_slots=data["n_time_slots"],
        alpha=data["alpha"],
        beta=data["beta"],
        gamma=data["gamma"],
    )
