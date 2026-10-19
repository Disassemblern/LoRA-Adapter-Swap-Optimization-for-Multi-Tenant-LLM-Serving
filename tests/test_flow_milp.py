"""
Hand-checked optima for the flow model. Units: seconds and requests.
Load-wait coefficient per load-second is (1 + demand * disk_s / slot_s).
"""
import numpy as np
import pytest

from src.cluster.flow_milp import solve
from src.cluster.types import AdapterSpec, FlowInstance, GpuSpec, LoadCosts

CFG = {"time_limit_sec": 30, "mip_gap": 0.0, "threads": 1, "log_to_console": False}
LOAD = LoadCosts(cpu_to_gpu_s=1.0, disk_to_gpu_s=5.0)
COEF = 1 + 1 * 5.0 / 60.0  # one request per slot


def _alternating(cpu_slots, **state):
    # adapters A, B alternate one request per slot on a 1-slot GPU
    return FlowInstance(
        adapters=(AdapterSpec("A", 100.0), AdapterSpec("B", 100.0)),
        gpus=(GpuSpec(lora_slots=1, cpu_slots=cpu_slots, tokens_per_slot=1000.0),),
        demand=((1, 0, 1, 0), (0, 1, 0, 1)),
        load=LOAD,
        **state,
    )


def test_alternating_demand_with_cpu_cache():
    plan = solve(_alternating(cpu_slots=2), CFG)
    # 4 GPU loads; each adapter comes from disk once. B's disk read is done at
    # t=0, when nobody is waiting for B, so it costs 4 s instead of 4 s * COEF.
    assert plan.gpu_loads == 4
    assert plan.cpu_inserts == 2
    assert plan.objective_s == pytest.approx((4 * 1.0 + 4.0) * COEF + 4.0)
    assert sum(plan.backlog.values()) == pytest.approx(0.0)


def test_alternating_demand_without_cpu_room_reloads_from_disk():
    plan = solve(_alternating(cpu_slots=1), CFG)
    assert plan.gpu_loads == 4
    assert plan.cpu_inserts == 4
    assert plan.objective_s == pytest.approx(4 * 5.0 * COEF)


def test_warm_start_skips_first_load():
    warm = frozenset({(0, 0)})
    plan = solve(_alternating(cpu_slots=2, gpu_loaded=warm, cpu_loaded=warm), CFG)
    assert plan.gpu_loads == 3
    assert plan.cpu_inserts == 1
    assert plan.objective_s == pytest.approx(3 * 1.0 * COEF + 4.0)  # B prefetched at t=0


def test_capacity_overflow_becomes_backlog():
    inst = FlowInstance(
        adapters=(AdapterSpec("A", 100.0),),
        gpus=(GpuSpec(lora_slots=1, cpu_slots=1, tokens_per_slot=200.0),),  # 2 req/slot
        demand=((3, 0),),
        load=LOAD,
        gpu_loaded=frozenset({(0, 0)}),
        cpu_loaded=frozenset({(0, 0)}),
    )
    plan = solve(inst, CFG)
    assert plan.backlog[(0, 0)] == pytest.approx(1.0)
    assert plan.backlog[(0, 1)] == pytest.approx(0.0)
    assert plan.objective_s == pytest.approx(60.0)  # one request waits one slot


def test_slot_limits_hold():
    rng = np.random.default_rng(0)
    inst = FlowInstance(
        adapters=tuple(AdapterSpec(f"a{i}", 500.0) for i in range(6)),
        gpus=(GpuSpec(2, 3, 5000.0), GpuSpec(2, 3, 5000.0)),
        demand=tuple(tuple(rng.integers(0, 4, size=4).tolist()) for _ in range(6)),
        load=LOAD,
    )
    plan = solve(inst, CFG)
    for j, g in enumerate(inst.gpus):
        for t in range(inst.horizon):
            assert sum((a, j, t) in plan.gpu_loaded for a in range(6)) <= g.lora_slots
            assert sum((a, j, t) in plan.cpu_loaded for a in range(6)) <= g.cpu_slots
    for (a, j, t), x in plan.served.items():
        if x > 1e-6:
            assert (a, j, t) in plan.gpu_loaded


def test_default_rolling_tick_fits_restricted_license():
    # 25 adapters, 4 GPUs, 3-slot horizon: the online controller's default size
    rng = np.random.default_rng(1)
    inst = FlowInstance(
        adapters=tuple(AdapterSpec(f"a{i}", 800.0) for i in range(25)),
        gpus=tuple(GpuSpec(4, 8, 60_000.0) for _ in range(4)),
        demand=tuple(tuple(rng.poisson(3, size=3).tolist()) for _ in range(25)),
        load=LOAD,
    )
    plan = solve(inst, {**CFG, "time_limit_sec": 20, "mip_gap": 0.01})
    assert plan.stats["n_vars"] < 2000
    assert plan.stats["n_constrs"] < 2000
