"""Inputs and outputs of the aggregated (flow) model, in physical units."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    tokens_per_request: float  # mean prompt + output tokens of this tenant (w_a)
    slo_weight: float = 1.0    # waiting-time weight: interactive > batch


@dataclass(frozen=True)
class GpuSpec:
    lora_slots: int         # SGLang --max-loras-per-batch (S_j)
    cpu_slots: int          # SGLang --max-loaded-loras (C_j), >= lora_slots
    tokens_per_slot: float  # token throughput per time slot (K_j)


@dataclass(frozen=True)
class LoadCosts:
    cpu_to_gpu_s: float   # adapter already in CPU memory -> GPU pool
    disk_to_gpu_s: float  # adapter on disk -> CPU memory -> GPU pool


@dataclass(frozen=True)
class FlowInstance:
    adapters: tuple[AdapterSpec, ...]
    gpus: tuple[GpuSpec, ...]
    demand: tuple[tuple[float, ...], ...]  # demand[a][t] = requests arriving in slot t
    load: LoadCosts
    slot_s: float = 60.0
    # state at the start of the horizon (rolling horizon); empty = cold start
    gpu_loaded: frozenset[tuple[int, int]] = frozenset()  # (adapter, gpu) in GPU pool
    cpu_loaded: frozenset[tuple[int, int]] = frozenset()  # (adapter, gpu) in CPU memory
    backlog: tuple[float, ...] = ()                         # waiting requests per adapter

    @property
    def horizon(self) -> int:
        return len(self.demand[0]) if self.demand else 0


@dataclass(frozen=True)
class FlowPlan:
    gpu_loaded: frozenset[tuple[int, int, int]]   # (adapter, gpu, t) in GPU pool
    cpu_loaded: frozenset[tuple[int, int, int]]   # (adapter, gpu, t) in CPU memory
    served: dict[tuple[int, int, int], float]     # (adapter, gpu, t) -> requests
    backlog: dict[tuple[int, int], float]         # (adapter, t) -> waiting requests
    gpu_loads: int       # loads into a GPU pool
    cpu_inserts: int     # adapters brought from disk into CPU memory
    objective_s: float   # weighted waiting, request-seconds
    status: str
    mip_gap: float
    wall_time_s: float
    stats: dict = field(default_factory=dict)  # model size, for license checks
