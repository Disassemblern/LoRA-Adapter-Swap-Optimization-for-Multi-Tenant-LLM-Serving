from src.data_types import Adapter, GPU, Request, Instance
from src.baselines.simulated_annealing import SimulatedAnnealing
from src.solution import verify


def _small_instance() -> Instance:
    adapters = [
        Adapter(id=0, memory_gb=1.0, swap_cost=1.0),
        Adapter(id=1, memory_gb=1.0, swap_cost=1.0),
    ]
    gpus = [GPU(id=0, vram_gb=20.0, base_model_gb=14.0, max_throughput=5)]
    requests = [
        Request(id=0, adapter_id=0, arrival_t=0, priority=2, latency_per_gpu=(1.0,)),
        Request(id=1, adapter_id=1, arrival_t=1, priority=1, latency_per_gpu=(1.0,)),
        Request(id=2, adapter_id=0, arrival_t=2, priority=3, latency_per_gpu=(1.0,)),
    ]
    return Instance(adapters=adapters, gpus=gpus, requests=requests,
                    n_time_slots=4, alpha=1.0, beta=2.0, gamma=0.5)


def test_sa_passes_verification():
    inst = _small_instance()
    sol = SimulatedAnnealing({"n_iterations": 500, "seed": 0}).solve(inst)
    violations = verify(sol, inst)
    assert violations == [], violations


def test_sa_all_requests_assigned():
    inst = _small_instance()
    sol = SimulatedAnnealing({"n_iterations": 500, "seed": 0}).solve(inst)
    assert set(sol.assignments.keys()) == {r.id for r in inst.requests}


def test_sa_objective_is_finite():
    inst = _small_instance()
    sol = SimulatedAnnealing({"n_iterations": 500, "seed": 0}).solve(inst)
    assert sol.objective < 1e5


def test_sa_counts_one_swap_for_adapter_kept_loaded():
    # Adapter 0 is needed in slots 0, 1 and 2 on the only GPU and always fits,
    # so it is loaded once and stays: exactly one swap.
    adapters = [Adapter(id=0, memory_gb=1.0, swap_cost=1.0)]
    gpus = [GPU(id=0, vram_gb=20.0, base_model_gb=14.0, max_throughput=5)]
    requests = [
        Request(id=t, adapter_id=0, arrival_t=t, priority=1, latency_per_gpu=(1.0,))
        for t in range(3)
    ]
    inst = Instance(adapters=adapters, gpus=gpus, requests=requests,
                    n_time_slots=3, alpha=1.0, beta=2.0, gamma=0.5)
    sol = SimulatedAnnealing({"n_iterations": 50, "seed": 0}).solve(inst)
    assert sum(sol.swaps.values()) == 1
