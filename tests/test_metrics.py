from src.data_types import Adapter, GPU, Request, Instance
from src.metrics import decompose
from src.solution import Solution


def test_swap_cost_is_derived_from_schedule_not_flags():
    adapters = [Adapter(id=0, memory_gb=1.0, swap_cost=3.0)]
    gpus = [GPU(id=0, vram_gb=20.0, base_model_gb=14.0, max_throughput=5)]
    requests = [Request(id=0, adapter_id=0, arrival_t=0, priority=1,
                        latency_per_gpu=(1.0,))]
    inst = Instance(adapters=adapters, gpus=gpus, requests=requests,
                    n_time_slots=3, alpha=1.0, beta=2.0, gamma=0.5)
    sol = Solution(
        assignments={0: 0},
        adapter_schedule={(0, 0, 0): True, (0, 0, 1): True, (0, 0, 2): True},
        swaps={(0, 0, 0): True, (0, 0, 1): True, (0, 0, 2): True},  # over-reported
        objective=0.0, mip_gap=0.0, wall_time_sec=0.0, status="TEST",
    )
    assert decompose(sol, inst)["swap"] == 3.0  # one load of a 3.0-cost adapter
