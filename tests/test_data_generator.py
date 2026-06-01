import pytest
from src import data_generator

BASE_CFG = {
    "instance": {
        "n_adapters": 5, "n_gpus": 2, "n_requests": 20, "n_time_slots": 8,
        "adapter_memory_gb": {"min": 0.5, "max": 3.0},
        "vram_gb": {"min": 24, "max": 40},
        "base_model_gb": 14.0,
        "request_priorities": [1, 2, 3], "priority_weights": [0.2, 0.5, 0.3],
        "arrival": "zipf", "zipf_alpha": 1.2, "seed": 42,
    },
    "objective": {"alpha": 1.0, "beta": 2.0, "gamma": 0.5},
}


def test_determinism():
    inst_a = data_generator.generate(BASE_CFG, seed=42)
    inst_b = data_generator.generate(BASE_CFG, seed=42)
    assert [a.memory_gb for a in inst_a.adapters] == [a.memory_gb for a in inst_b.adapters]
    assert [r.adapter_id for r in inst_a.requests] == [r.adapter_id for r in inst_b.requests]


def test_different_seeds_differ():
    inst_a = data_generator.generate(BASE_CFG, seed=1)
    inst_b = data_generator.generate(BASE_CFG, seed=2)
    assert [r.adapter_id for r in inst_a.requests] != [r.adapter_id for r in inst_b.requests]


def test_correct_counts():
    inst = data_generator.generate(BASE_CFG, seed=42)
    assert len(inst.adapters) == 5
    assert len(inst.gpus) == 2
    assert len(inst.requests) == 20
    assert inst.n_time_slots == 8


def test_arrival_times_in_range():
    inst = data_generator.generate(BASE_CFG, seed=42)
    for req in inst.requests:
        assert 0 <= req.arrival_t < inst.n_time_slots


def test_feasibility_check_raises():
    import copy
    cfg = copy.deepcopy(BASE_CFG)
    cfg["instance"]["n_requests"] = 10_000  # impossible to schedule
    with pytest.raises(ValueError, match="infeasible"):
        data_generator.generate(cfg, seed=42)
