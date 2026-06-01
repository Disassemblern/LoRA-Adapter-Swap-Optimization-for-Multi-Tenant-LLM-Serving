"""
Solve a tiny hand-designed instance where we know roughly what to expect,
then confirm the solution is valid and Gurobi found an optimal result.
"""
from src.data_types import Adapter, GPU, Request, Instance
from src import model_gurobi
from src.solution import verify

SOLVER_CFG = {
    "time_limit_sec": 60,
    "mip_gap": 0.001,
    "threads": 1,
    "log_to_console": False,
}


def _toy_instance():
    adapters = [
        Adapter(id=0, memory_gb=1.0, swap_cost=1.0),
        Adapter(id=1, memory_gb=1.0, swap_cost=1.0),
        Adapter(id=2, memory_gb=1.0, swap_cost=1.0),
    ]
    gpus = [
        GPU(id=0, vram_gb=20.0, base_model_gb=14.0, max_throughput=5),
        GPU(id=1, vram_gb=20.0, base_model_gb=14.0, max_throughput=5),
    ]
    requests = [
        Request(id=0, adapter_id=0, arrival_t=0, priority=1, latency_per_gpu=(1.0, 2.0)),
        Request(id=1, adapter_id=0, arrival_t=1, priority=2, latency_per_gpu=(1.0, 2.0)),
        Request(id=2, adapter_id=1, arrival_t=0, priority=1, latency_per_gpu=(1.0, 2.0)),
        Request(id=3, adapter_id=2, arrival_t=1, priority=3, latency_per_gpu=(1.0, 2.0)),
        Request(id=4, adapter_id=2, arrival_t=2, priority=2, latency_per_gpu=(1.0, 2.0)),
    ]
    return Instance(adapters=adapters, gpus=gpus, requests=requests,
                    n_time_slots=3, alpha=1.0, beta=2.0, gamma=0.5)


def test_gurobi_finds_optimal():
    inst = _toy_instance()
    sol = model_gurobi.build_and_solve(inst, SOLVER_CFG)
    assert sol.status == "OPTIMAL"
    assert sol.mip_gap < 0.01


def test_gurobi_solution_passes_verification():
    inst = _toy_instance()
    sol = model_gurobi.build_and_solve(inst, SOLVER_CFG)
    violations = verify(sol, inst)
    assert violations == [], violations


def test_gurobi_assigns_all_requests():
    inst = _toy_instance()
    sol = model_gurobi.build_and_solve(inst, SOLVER_CFG)
    assert set(sol.assignments.keys()) == {r.id for r in inst.requests}


def test_gurobi_prefers_low_latency_gpu():
    # GPU 0 has latency 1.0, GPU 1 has latency 2.0.
    # Optimal should assign everything to GPU 0.
    inst = _toy_instance()
    sol = model_gurobi.build_and_solve(inst, SOLVER_CFG)
    for req in inst.requests:
        assert sol.assignments[req.id] == 0, (
            f"Request {req.id} went to GPU {sol.assignments[req.id]}, expected GPU 0"
        )
