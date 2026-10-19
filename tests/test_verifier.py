"""
Each test constructs a minimal valid solution then breaks exactly one constraint
to confirm the verifier catches it.
"""
from src.data_types import Adapter, GPU, Request, Instance
from src.solution import Solution, verify


def _tiny_instance():
    adapters = [Adapter(id=0, memory_gb=1.0, swap_cost=1.0)]
    gpus = [GPU(id=0, vram_gb=20.0, base_model_gb=14.0, max_throughput=2)]
    requests = [Request(id=0, adapter_id=0, arrival_t=0, priority=1,
                        latency_per_gpu=(1.0,))]
    return Instance(adapters=adapters, gpus=gpus, requests=requests,
                    n_time_slots=3, alpha=1.0, beta=2.0, gamma=0.5)


def _valid_solution(inst):
    return Solution(
        assignments={0: 0},
        adapter_schedule={(0, 0, 0): True, (0, 0, 1): False, (0, 0, 2): False},
        swaps={(0, 0, 0): True, (0, 0, 1): False, (0, 0, 2): False},
        objective=0.0, mip_gap=0.0, wall_time_sec=0.0, status="TEST",
    )


def test_valid_solution_passes():
    inst = _tiny_instance()
    assert verify(_valid_solution(inst), inst) == []


def test_missing_assignment_detected():
    inst = _tiny_instance()
    sol = _valid_solution(inst)
    sol.assignments.clear()
    violations = verify(sol, inst)
    assert any("no GPU assignment" in v for v in violations)


def test_adapter_not_loaded_detected():
    inst = _tiny_instance()
    sol = _valid_solution(inst)
    sol.adapter_schedule[(0, 0, 0)] = False  # adapter not loaded at arrival time
    violations = verify(sol, inst)
    assert any("not loaded" in v for v in violations)


def test_memory_overflow_detected():
    adapters = [
        Adapter(id=0, memory_gb=4.0, swap_cost=4.0),
        Adapter(id=1, memory_gb=4.0, swap_cost=4.0),
    ]
    gpus = [GPU(id=0, vram_gb=20.0, base_model_gb=14.0, max_throughput=5)]
    requests = [Request(id=0, adapter_id=0, arrival_t=0, priority=1,
                        latency_per_gpu=(1.0,))]
    inst = Instance(adapters=adapters, gpus=gpus, requests=requests,
                    n_time_slots=1, alpha=1.0, beta=2.0, gamma=0.5)

    # Both adapters loaded simultaneously: 4+4=8 GB > budget of 6 GB
    sol = Solution(
        assignments={0: 0},
        adapter_schedule={(0, 0, 0): True, (1, 0, 0): True},
        swaps={(0, 0, 0): True, (1, 0, 0): True},
        objective=0.0, mip_gap=0.0, wall_time_sec=0.0, status="TEST",
    )
    violations = verify(sol, inst)
    assert any("GB used >" in v for v in violations)


def test_throughput_exceeded_detected():
    inst = _tiny_instance()
    # Add a second request at the same time slot on the same GPU
    # max_throughput=2, so 3 requests should fail
    from src.data_types import Request
    extra_reqs = [
        Request(id=0, adapter_id=0, arrival_t=0, priority=1, latency_per_gpu=(1.0,)),
        Request(id=1, adapter_id=0, arrival_t=0, priority=1, latency_per_gpu=(1.0,)),
        Request(id=2, adapter_id=0, arrival_t=0, priority=1, latency_per_gpu=(1.0,)),
    ]
    inst2 = Instance(adapters=inst.adapters, gpus=inst.gpus, requests=extra_reqs,
                     n_time_slots=3, alpha=1.0, beta=2.0, gamma=0.5)
    sol = Solution(
        assignments={0: 0, 1: 0, 2: 0},
        adapter_schedule={(0, 0, 0): True, (0, 0, 1): False, (0, 0, 2): False},
        swaps={(0, 0, 0): True, (0, 0, 1): False, (0, 0, 2): False},
        objective=0.0, mip_gap=0.0, wall_time_sec=0.0, status="TEST",
    )
    violations = verify(sol, inst2)
    assert any("max_throughput" in v for v in violations)


def test_unreported_swap_detected():
    inst = _tiny_instance()
    sol = _valid_solution(inst)
    sol.swaps[(0, 0, 0)] = False  # adapter loaded at t=0 but no swap reported
    violations = verify(sol, inst)
    assert any("swap" in v for v in violations)
