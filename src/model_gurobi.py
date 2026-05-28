"""
Mathematical formulation:

Decision variables:
  x[i, j]      = 1 if request i is assigned to GPU j
  y[a, j, t]   = 1 if adapter a is loaded on GPU j at time slot t
  s[a, j, t]   = 1 if adapter a is swapped IN on GPU j at time slot t
  delay[i]     >= 0, continuous — effective latency of request i

Objective (minimize):
  alpha * sum(latency[i,j] * x[i,j])          -- total serving latency
  + beta  * sum(swap_cost[a] * s[a,j,t])      -- total swap overhead
  + gamma * sum(priority[i] * delay[i])        -- weighted priority delay

Constraints:
  (1) Assignment:      sum_j x[i,j] = 1                      for all i
  (2) Availability:    x[i,j] <= y[adapter(i), j, arrival(i)] for all i, j
  (3) Memory:          sum_a memory[a]*y[a,j,t] <= budget[j]  for all j, t
  (4) Swap detection:  s[a,j,t] >= y[a,j,t] - y[a,j,t-1]    for all a, j, t>=1
                       s[a,j,0] >= y[a,j,0]                  (all unloaded at t=0)
  (5) Throughput:      sum_{i: arrival==t} x[i,j] <= max_tp[j] for all j, t
  (6) Delay linkage:   delay[i] >= latency[i,j] * x[i,j]     for all i, j
"""
from __future__ import annotations
import time

import gurobipy as gp
from gurobipy import GRB

from src.data_types import Instance
from src.solution import Solution


def build_and_solve(inst: Instance, solver_cfg: dict) -> Solution:
    m = gp.Model("lora_swap")
    _apply_solver_cfg(m, solver_cfg)

    x, y, s, delay = _add_variables(m, inst)
    _add_assignment_constraints(m, inst, x)
    _add_availability_constraints(m, inst, x, y)
    _add_memory_constraints(m, inst, y)
    _add_swap_detection_constraints(m, inst, y, s)
    _add_throughput_constraints(m, inst, x)
    _add_delay_linkage(m, inst, x, delay)
    _set_objective(m, inst, x, s, delay)

    t0 = time.time()
    m.optimize()
    wall_time = time.time() - t0

    return _extract_solution(m, inst, x, y, s, wall_time)


# solver config 

def _apply_solver_cfg(m: gp.Model, cfg: dict) -> None:
    m.Params.TimeLimit = cfg["time_limit_sec"]
    m.Params.MIPGap = cfg["mip_gap"]
    m.Params.Threads = cfg["threads"]
    m.Params.LogToConsole = int(cfg["log_to_console"])


# variables
def _add_variables(
    m: gp.Model, inst: Instance
) -> tuple[gp.tupledict, gp.tupledict, gp.tupledict, gp.tupledict]:
    # x[i, j]: binary — request i assigned to GPU j
    x = m.addVars(
        [(req.id, g.id) for req in inst.requests for g in inst.gpus],
        vtype=GRB.BINARY,
        name="x",
    )

    # y[a, j, t]: binary — adapter a loaded on GPU j at time t
    y = m.addVars(
        [(a.id, g.id, t)
         for a in inst.adapters
         for g in inst.gpus
         for t in range(inst.n_time_slots)],
        vtype=GRB.BINARY,
        name="y",
    )

    # s[a, j, t]: binary — adapter a swapped IN on GPU j at time t
    s = m.addVars(
        [(a.id, g.id, t)
         for a in inst.adapters
         for g in inst.gpus
         for t in range(inst.n_time_slots)],
        vtype=GRB.BINARY,
        name="s",
    )

    # delay[i]: continuous — effective latency experienced by request i
    delay = m.addVars(
        [req.id for req in inst.requests],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="delay",
    )

    return x, y, s, delay


# constraints

def _add_assignment_constraints(m: gp.Model, inst: Instance, x: gp.tupledict) -> None:
    # (1) Each request must go to exactly one GPU:
    #     sum_j x[i, j] = 1   for all i
    for req in inst.requests:
        m.addConstr(
            gp.quicksum(x[req.id, g.id] for g in inst.gpus) == 1,
            name=f"assign_{req.id}",
        )


def _add_availability_constraints(
    m: gp.Model, inst: Instance, x: gp.tupledict, y: gp.tupledict
) -> None:
    # (2) A request can only go to a GPU that has its adapter loaded:
    #     x[i, j] <= y[adapter(i), j, arrival(i)]   for all i, j
    for req in inst.requests:
        for g in inst.gpus:
            m.addConstr(
                x[req.id, g.id] <= y[req.adapter_id, g.id, req.arrival_t],
                name=f"avail_{req.id}_{g.id}",
            )


def _add_memory_constraints(
    m: gp.Model, inst: Instance, y: gp.tupledict
) -> None:
    # (3) Total memory of loaded adapters cannot exceed the GPU's adapter budget:
    #     sum_a memory[a] * y[a, j, t] <= vram[j] - base_model   for all j, t
    for g in inst.gpus:
        for t in range(inst.n_time_slots):
            m.addConstr(
                gp.quicksum(a.memory_gb * y[a.id, g.id, t] for a in inst.adapters)
                <= g.adapter_budget_gb,
                name=f"mem_{g.id}_{t}",
            )


def _add_swap_detection_constraints(
    m: gp.Model, inst: Instance, y: gp.tupledict, s: gp.tupledict
) -> None:
    # (4) Detect when an adapter goes from unloaded -> loaded (a swap-in event):
    #     s[a, j, t] >= y[a, j, t] - y[a, j, t-1]   for t >= 1
    #     s[a, j, 0] >= y[a, j, 0]                   (GPU starts empty at t=0)
    for a in inst.adapters:
        for g in inst.gpus:
            # t=0: no previous state, so any loaded adapter counts as a swap
            m.addConstr(
                s[a.id, g.id, 0] >= y[a.id, g.id, 0],
                name=f"swap0_{a.id}_{g.id}",
            )
            for t in range(1, inst.n_time_slots):
                m.addConstr(
                    s[a.id, g.id, t] >= y[a.id, g.id, t] - y[a.id, g.id, t - 1],
                    name=f"swap_{a.id}_{g.id}_{t}",
                )


def _add_throughput_constraints(
    m: gp.Model, inst: Instance, x: gp.tupledict
) -> None:
    # (5) Each GPU can only process up to max_throughput requests per time slot:
    #     sum_{i: arrival(i)==t} x[i, j] <= max_throughput[j]   for all j, t
    for g in inst.gpus:
        for t in range(inst.n_time_slots):
            reqs_at_t = [req for req in inst.requests if req.arrival_t == t]
            if not reqs_at_t:
                continue
            m.addConstr(
                gp.quicksum(x[req.id, g.id] for req in reqs_at_t) <= g.max_throughput,
                name=f"tp_{g.id}_{t}",
            )


def _add_delay_linkage(
    m: gp.Model, inst: Instance, x: gp.tupledict, delay: gp.tupledict
) -> None:
    # (6) delay[i] captures the latency of whichever GPU handles request i:
    #     delay[i] >= latency[i, j] * x[i, j]   for all i, j
    # Because we minimize gamma * priority[i] * delay[i], this pushes
    # high-priority requests toward lower-latency GPUs.
    for req in inst.requests:
        for g in inst.gpus:
            m.addConstr(
                delay[req.id] >= req.latency_per_gpu[g.id] * x[req.id, g.id],
                name=f"delay_{req.id}_{g.id}",
            )


# objective

def _set_objective(
    m: gp.Model,
    inst: Instance,
    x: gp.tupledict,
    s: gp.tupledict,
    delay: gp.tupledict,
) -> None:
    # Minimize: alpha*latency + beta*swap_cost + gamma*priority_delay
    latency_term = gp.quicksum(
        req.latency_per_gpu[g.id] * x[req.id, g.id]
        for req in inst.requests
        for g in inst.gpus
    )
    swap_term = gp.quicksum(
        a.swap_cost * s[a.id, g.id, t]
        for a in inst.adapters
        for g in inst.gpus
        for t in range(inst.n_time_slots)
    )
    priority_term = gp.quicksum(
        req.priority * delay[req.id] for req in inst.requests
    )

    m.setObjective(
        inst.alpha * latency_term + inst.beta * swap_term + inst.gamma * priority_term,
        GRB.MINIMIZE,
    )


# solution extraction

def _extract_solution(
    m: gp.Model,
    inst: Instance,
    x: gp.tupledict,
    y: gp.tupledict,
    s: gp.tupledict,
    wall_time: float,
) -> Solution:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    status = status_map.get(m.Status, f"GUROBI_{m.Status}")

    if m.SolCount == 0:
        raise RuntimeError(
            f"Gurobi found no feasible solution (status={status}). "
        )

    assignments = {
        req.id: g.id
        for req in inst.requests
        for g in inst.gpus
        if x[req.id, g.id].X > 0.5
    }

    adapter_schedule = {
        (a.id, g.id, t): (y[a.id, g.id, t].X > 0.5)
        for a in inst.adapters
        for g in inst.gpus
        for t in range(inst.n_time_slots)
    }

    swaps = {
        (a.id, g.id, t): (s[a.id, g.id, t].X > 0.5)
        for a in inst.adapters
        for g in inst.gpus
        for t in range(inst.n_time_slots)
    }

    return Solution(
        assignments=assignments,
        adapter_schedule=adapter_schedule,
        swaps=swaps,
        objective=m.ObjVal,
        mip_gap=m.MIPGap,
        wall_time_sec=wall_time,
        status=status,
    )
