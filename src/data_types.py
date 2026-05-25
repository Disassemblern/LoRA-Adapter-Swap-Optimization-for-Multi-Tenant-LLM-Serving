from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class Adapter: # one LoRA adapter
    id: int
    memory_gb: float # size of the adapter
    swap_cost: float  # precomputed as k * memory_gb

@dataclass(frozen=True)
class GPU: # one GPU
    id: int
    vram_gb: float # size of the gpu
    base_model_gb: float # allocated vram for the base model
    max_throughput: int  # max requests this GPU can handle per time slot

    @property
    def adapter_budget_gb(self) -> float:
        #Free VRAM available for adapters after the base model is loaded.
        return self.vram_gb - self.base_model_gb

@dataclass(frozen=True)
class Request: # one user request
    id: int
    adapter_id: int # which adapter it needs
    arrival_t: int  # which time slot this request arrives in
    priority: int   # 1=low, 2=medium, 3=high
    latency_per_gpu: tuple[float, ...]  # latency[j] = cost if handled by GPU j

@dataclass(frozen=True)
class Instance: # the entire problem
    adapters: Sequence[Adapter]
    gpus: Sequence[GPU]
    requests: Sequence[Request]
    n_time_slots: int
    alpha: float  # objective weight for latency
    beta: float   # objective weight for swap cost
    gamma: float  # objective weight for priority delay
