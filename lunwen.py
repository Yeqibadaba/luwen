from pathlib import Path

code = r'''"""
SN-NSGA-II for robust collaborative scheduling of reach stackers and trucks in a dry port.

This implementation is generated from the uploaded thesis drafts and follows the documented design:
1) three-segment chromosome:
   - rail reach stacker assignment
   - yard reach stacker + truck assignment
   - global task priority permutation
2) decoder-based scheduling with hard-constraint oriented repair
3) two objectives:
   - f1: makespan
   - f2: total lifecycle cost = baseline energy + time-window penalties + beta * CVaR(failure_loss)
4) nested Monte Carlo simulation for equipment failure risk evaluation
5) improved NSGA-II:
   - constraint-domination
   - elitist archive
   - heuristic + random initialization
   - segment-wise crossover / mutation

Important note
--------------
The uploaded text contains several mathematical expressions that are truncated in the extracted text, so this
program implements a document-consistent, experiment-ready engineering version of the model rather than a strict
symbol-by-symbol reproduction of every omitted formula. All key mechanisms requested by the documents are included,
and all experiment parameters are centralized for easy modification.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

# ============================================================
# 1. Data structures
# ============================================================

TASK_TYPES_INTERNAL = {"A", "C", "E"}
TASK_TYPES_EXTERNAL = {"B", "D"}
TASK_TYPES_NEED_YARD = {"A", "C", "E"}


@dataclass
class TimeWindow:
    start: float
    end: float


@dataclass
class Task:
    task_id: int
    task_type: str  # A/B/C/D/E
    rail_point_id: int
    rail_point_xy: Tuple[float, float]
    yard_zone_id: Optional[int] = None
    yard_zone_xy: Optional[Tuple[float, float]] = None
    train_service_time: float = 8.0
    yard_service_time: float = 6.0
    arrival_window: Optional[TimeWindow] = None  # only B/D usually
    max_train_delay: float = 2.0   # uncertainty for first-layer robust buffer
    max_yard_delay: float = 1.5    # uncertainty for first-layer robust buffer

    def requires_yard(self) -> bool:
        return self.task_type in TASK_TYPES_NEED_YARD

    def requires_internal_truck(self) -> bool:
        return self.task_type in TASK_TYPES_INTERNAL

    def requires_external_truck(self) -> bool:
        return self.task_type in TASK_TYPES_EXTERNAL


@dataclass
class EquipmentConfig:
    n_train_rs: int = 2
    n_yard_rs: int = 2
    n_internal_trucks: int = 6
    n_external_trucks: int = 3


@dataclass
class EnergyParams:
    electricity_price: float = 0.9
    fuel_price: float = 7.2

    train_op_power: float = 55.0
    yard_op_power: float = 50.0
    train_move_power: float = 20.0
    yard_move_power: float = 18.0
    train_idle_power: float = 10.0
    yard_idle_power: float = 8.0

    truck_idle_fuel_per_hour: float = 2.0
    truck_shutdown_idle_power: float = 0.1
    truck_loaded_fuel_per_km: float = 0.45
    truck_empty_fuel_per_km: float = 0.30

    early_penalty_per_hour: float = 20.0
    late_penalty_per_hour: float = 60.0
    failure_idle_energy_cost_per_hour: float = 15.0


@dataclass
class SpeedParams:
    train_rs_speed_kmph: float = 12.0
    yard_rs_speed_kmph: float = 10.0
    truck_loaded_speed_kmph: float = 18.0
    truck_empty_speed_kmph: float = 22.0


@dataclass
class RobustParams:
    gamma_1: int = 3           # first-layer robust budget
    beta: float = 1.0          # failure risk aversion
    alpha: float = 0.95        # CVaR confidence level
    ext_early_slack: float = 0.5  # hours
    ext_late_slack: float = 0.75  # hours


@dataclass
class FailureParams:
    # failures per hour
    lambda_train_rs: float = 0.015
    lambda_yard_rs: float = 0.012
    lambda_truck: float = 0.010
    repair_time_mean_train_rs: float = 0.8
    repair_time_mean_yard_rs: float = 0.7
    repair_time_mean_truck: float = 0.5
    gamma_reschedule: float = 0.1   # time-energy tradeoff in fast rescheduling
    monte_carlo_scenarios: int = 50
    common_random_seed: int = 20260421


@dataclass
class DecoderParams:
    t1_idle_shutdown_threshold: float = 0.20  # hours
    t2_truck_shutdown_threshold: float = 0.15 # hours
    nmax_start_stop: int = 20
    big_m: float = 1e6


@dataclass
class GAParams:
    population_size: int = 100
    n_generations: int = 80
    crossover_prob: float = 0.85
    mutation_prob: float = 0.08
    tournament_size: int = 2
    archive_max_size: int = 200
    max_no_improve_generations: int = 25
    heuristic_ratio: float = 0.30
    seed: int = 42


@dataclass
class ProblemInstance:
    name: str
    tasks: List[Task]
    yard_zones_xy: Dict[int, Tuple[float, float]]
    gate_xy: Tuple[float, float]
    equipment: EquipmentConfig
    energy: EnergyParams
    speed: SpeedParams
    robust: RobustParams
    failure: FailureParams
    decoder: DecoderParams


# ============================================================
# 2. Chromosome and schedule structures
# ============================================================

@dataclass
class Chromosome:
    train_assign: List[int]  # length N, values in [0, n_train_rs-1]
    truck_assign: List[int]  # length N, values in [0, n_trucks-1] with separate id spaces handled by decoder
    yard_assign: List[int]   # length N, values in [-1, n_yard_rs-1], B/D forced to -1
    priority: List[int]      # permutation of task indices [0..N-1]

    def clone(self) -> "Chromosome":
        return Chromosome(
            train_assign=self.train_assign[:],
            truck_assign=self.truck_assign[:],
            yard_assign=self.yard_assign[:],
            priority=self.priority[:],
        )


@dataclass
class TaskScheduleRecord:
    task_id: int
    train_rs_id: int
    truck_id: int
    yard_rs_id: Optional[int]
    train_start: float = 0.0
    train_finish: float = 0.0
    truck_line_arrive: float = 0.0
    truck_line_leave: float = 0.0
    yard_arrive: Optional[float] = None
    yard_finish: Optional[float] = None
    early: float = 0.0
    late: float = 0.0
    feasible: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class Schedule:
    task_records: Dict[int, TaskScheduleRecord]
    makespan: float
    baseline_energy_cost: float
    time_window_penalty: float
    objective_2_baseline: float
    feasible: bool
    constraint_violation: float
    device_sequences: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    chromosome: Chromosome
    feasible: bool
    constraint_violation: float
    f1_makespan: float
    f2_cost: float
    baseline_cost: float
    failure_cvar_cost: float
    schedule: Schedule


@dataclass
class FailureEvent:
    device_kind: str   # train_rs / yard_rs / truck
    device_id: int
    start_time: float
    duration: float


# ============================================================
# 3. Utility functions
# ============================================================

def manhattan_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def km_to_hours(distance_km: float, speed_kmph: float) -> float:
    if speed_kmph <= 0:
        raise ValueError("speed must be positive")
    return distance_km / speed_kmph


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


# ============================================================
# 4. Instance generation / loading
# ============================================================

def default_population_size(n_tasks: int) -> int:
    if n_tasks <= 50:
        return 100
    if n_tasks <= 200:
        return 200
    return 300


def build_default_instance(scale: str = "small", seed: int = 1) -> ProblemInstance:
    rng = random.Random(seed)

    if scale == "small":
        n_tasks = 40
        equipment = EquipmentConfig(n_train_rs=2, n_yard_rs=2, n_internal_trucks=8, n_external_trucks=3)
    elif scale == "medium":
        n_tasks = 100
        equipment = EquipmentConfig(n_train_rs=2, n_yard_rs=2, n_internal_trucks=6, n_external_trucks=3)
    elif scale == "large":
        n_tasks = 200
        equipment = EquipmentConfig(n_train_rs=2, n_yard_rs=2, n_internal_trucks=6, n_external_trucks=3)
    else:
        raise ValueError("scale must be small / medium / large")

    yard_zones_xy = {
        1: (1.0, 2.0),  # temporary storage
        2: (3.0, 4.0),  # long storage
        3: (4.0, 1.5),  # export zone
        4: (2.0, 5.0),  # empty container zone
    }
    gate_xy = (0.0, 0.0)

    # Type mix can be tuned later from experiments.
    task_type_pool = (
        ["A"] * int(n_tasks * 0.30) +
        ["B"] * int(n_tasks * 0.15) +
        ["C"] * int(n_tasks * 0.25) +
        ["D"] * int(n_tasks * 0.15)
    )
    while len(task_type_pool) < n_tasks:
        task_type_pool.append("E")
    rng.shuffle(task_type_pool)

    tasks: List[Task] = []
    for i in range(n_tasks):
        ttype = task_type_pool[i]
        rail_point_id = rng.randint(1, 25)
        rail_xy = (0.0, float(rail_point_id))
        yard_zone = None
        yard_xy = None

        if ttype in {"A", "E"}:
            yard_zone = 4 if ttype == "E" else rng.choice([1, 2])
            yard_xy = yard_zones_xy[yard_zone]
        elif ttype == "C":
            yard_zone = 3
            yard_xy = yard_zones_xy[yard_zone]

        tw = None
        if ttype in {"B", "D"}:
            a = rng.uniform(0, max(1.0, n_tasks / 20))
            width = rng.uniform(0.2, 1.2)
            tw = TimeWindow(start=a, end=a + width)

        task = Task(
            task_id=i,
            task_type=ttype,
            rail_point_id=rail_point_id,
            rail_point_xy=rail_xy,
            yard_zone_id=yard_zone,
            yard_zone_xy=yard_xy,
            train_service_time=rng.uniform(0.08, 0.22),
            yard_service_time=rng.uniform(0.06, 0.18),
            arrival_window=tw,
            max_train_delay=rng.uniform(0.01, 0.05),
            max_yard_delay=rng.uniform(0.01, 0.04),
        )
        tasks.append(task)

    instance = ProblemInstance(
        name=f"dry_port_{scale}",
        tasks=tasks,
        yard_zones_xy=yard_zones_xy,
        gate_xy=gate_xy,
        equipment=equipment,
        energy=EnergyParams(),
        speed=SpeedParams(),
        robust=RobustParams(),
        failure=FailureParams(),
        decoder=DecoderParams(),
    )
    return instance


# ============================================================
# 5. Chromosome generation and repair
# ============================================================

def total_trucks(instance: ProblemInstance) -> int:
    return instance.equipment.n_internal_trucks + instance.equipment.n_external_trucks


def internal_truck_ids(instance: ProblemInstance) -> List[int]:
    return list(range(instance.equipment.n_internal_trucks))


def external_truck_ids(instance: ProblemInstance) -> List[int]:
    start = instance.equipment.n_internal_trucks
    return list(range(start, start + instance.equipment.n_external_trucks))


def legal_truck_ids_for_task(task: Task, instance: ProblemInstance) -> List[int]:
    if task.requires_internal_truck():
        return internal_truck_ids(instance)
    return external_truck_ids(instance)


def legal_yard_ids_for_task(task: Task, instance: ProblemInstance) -> List[int]:
    if task.requires_yard():
        return list(range(instance.equipment.n_yard_rs))
    return [-1]


def repair_chromosome(ch: Chromosome, instance: ProblemInstance, rng: random.Random) -> Chromosome:
    n = len(instance.tasks)
    if len(ch.train_assign) != n or len(ch.truck_assign) != n or len(ch.yard_assign) != n or len(ch.priority) != n:
        raise ValueError("invalid chromosome length")

    # Repair priority permutation
    seen = set()
    fixed_priority = []
    missing = [i for i in range(n) if i not in ch.priority]
    miss_iter = iter(missing)
    for gene in ch.priority:
        if gene not in seen and 0 <= gene < n:
            fixed_priority.append(gene)
            seen.add(gene)
        else:
            fixed_priority.append(next(miss_iter))
    ch.priority = fixed_priority

    # Repair assignments
    for i, task in enumerate(instance.tasks):
        ch.train_assign[i] = max(0, min(instance.equipment.n_train_rs - 1, int(ch.train_assign[i])))

        legal_trucks = legal_truck_ids_for_task(task, instance)
        if ch.truck_assign[i] not in legal_trucks:
            ch.truck_assign[i] = rng.choice(legal_trucks)

        legal_yards = legal_yard_ids_for_task(task, instance)
        if ch.yard_assign[i] not in legal_yards:
            ch.yard_assign[i] = legal_yards[0] if len(legal_yards) == 1 else rng.choice(legal_yards)

    return ch


def heuristic_priority_edd(tasks: List[Task]) -> List[int]:
    def key(task: Task) -> Tuple[float, float]:
        due = task.arrival_window.end if task.arrival_window else float("inf")
        return (due, task.train_service_time)
    return [t.task_id for t in sorted(tasks, key=key)]


def heuristic_priority_spt(tasks: List[Task]) -> List[int]:
    return [t.task_id for t in sorted(tasks, key=lambda t: t.train_service_time)]


def heuristic_priority_energy(tasks: List[Task]) -> List[int]:
    def est_energy(task: Task) -> float:
        return task.train_service_time + (task.yard_service_time if task.requires_yard() else 0.0)
    return [t.task_id for t in sorted(tasks, key=est_energy)]


def heuristic_priority_robust(tasks: List[Task]) -> List[int]:
    return [t.task_id for t in sorted(tasks, key=lambda t: (t.max_train_delay + t.max_yard_delay))]


def make_heuristic_individual(instance: ProblemInstance, rule_name: str, rng: random.Random) -> Chromosome:
    tasks = instance.tasks
    n = len(tasks)
    if rule_name == "EDD":
        priority = heuristic_priority_edd(tasks)
    elif rule_name == "SPT":
        priority = heuristic_priority_spt(tasks)
    elif rule_name == "ENERGY":
        priority = heuristic_priority_energy(tasks)
    elif rule_name == "ROBUST":
        priority = heuristic_priority_robust(tasks)
    else:
        raise ValueError(rule_name)

    # Earliest available / rough balancing assignment
    train_load = [0.0] * instance.equipment.n_train_rs
    yard_load = [0.0] * instance.equipment.n_yard_rs
    truck_load = [0.0] * total_trucks(instance)

    train_assign = [0] * n
    truck_assign = [0] * n
    yard_assign = [-1] * n

    for tid in priority:
        task = tasks[tid]

        best_r = min(range(instance.equipment.n_train_rs), key=lambda r: train_load[r])
        train_assign[tid] = best_r
        train_load[best_r] += task.train_service_time + task.max_train_delay

        legal_trucks = legal_truck_ids_for_task(task, instance)
        best_k = min(legal_trucks, key=lambda k: truck_load[k])
        truck_assign[tid] = best_k
        truck_load[best_k] += task.train_service_time + 0.05

        if task.requires_yard():
            best_y = min(range(instance.equipment.n_yard_rs), key=lambda y: yard_load[y])
            yard_assign[tid] = best_y
            yard_load[best_y] += task.yard_service_time + task.max_yard_delay
        else:
            yard_assign[tid] = -1

    ch = Chromosome(train_assign=train_assign, truck_assign=truck_assign, yard_assign=yard_assign, priority=priority)
    return repair_chromosome(ch, instance, rng)


def make_random_individual(instance: ProblemInstance, rng: random.Random) -> Chromosome:
    n = len(instance.tasks)
    train_assign = [rng.randrange(instance.equipment.n_train_rs) for _ in range(n)]
    truck_assign = []
    yard_assign = []
    for task in instance.tasks:
        truck_assign.append(rng.choice(legal_truck_ids_for_task(task, instance)))
        yard_assign.append(rng.choice(legal_yard_ids_for_task(task, instance)))
    priority = list(range(n))
    rng.shuffle(priority)
    ch = Chromosome(train_assign=train_assign, truck_assign=truck_assign, yard_assign=yard_assign, priority=priority)
    return repair_chromosome(ch, instance, rng)


def initialize_population(instance: ProblemInstance, ga: GAParams, rng: random.Random) -> List[Chromosome]:
    pop = []
    heuristic_count = int(ga.population_size * ga.heuristic_ratio)
    rules = ["EDD", "SPT", "ENERGY", "ROBUST"]
    for i in range(heuristic_count):
        pop.append(make_heuristic_individual(instance, rules[i % len(rules)], rng))
    while len(pop) < ga.population_size:
        pop.append(make_random_individual(instance, rng))
    return pop


# ============================================================
# 6. Decoder
# ============================================================

def robust_buffer_for_train_task(task: Task) -> float:
    return task.max_train_delay


def robust_buffer_for_yard_task(task: Task) -> float:
    return task.max_yard_delay


def task_loaded_distance(task: Task, instance: ProblemInstance) -> float:
    if task.task_type in {"A", "E"}:
        return manhattan_distance(task.rail_point_xy, task.yard_zone_xy)
    if task.task_type == "B":
        return manhattan_distance(task.rail_point_xy, instance.gate_xy)
    if task.task_type == "C":
        return manhattan_distance(task.yard_zone_xy, task.rail_point_xy)
    if task.task_type == "D":
        return manhattan_distance(instance.gate_xy, task.rail_point_xy)
    raise ValueError(task.task_type)


def sequence_by_device(ch: Chromosome, instance: ProblemInstance) -> Dict[str, Dict[int, List[int]]]:
    seq = {
        "train_rs": {r: [] for r in range(instance.equipment.n_train_rs)},
        "yard_rs": {y: [] for y in range(instance.equipment.n_yard_rs)},
        "truck": {k: [] for k in range(total_trucks(instance))},
    }
    for tid in ch.priority:
        seq["train_rs"][ch.train_assign[tid]].append(tid)
        seq["truck"][ch.truck_assign[tid]].append(tid)
        if ch.yard_assign[tid] >= 0:
            seq["yard_rs"][ch.yard_assign[tid]].append(tid)
    return seq


def decode_chromosome(ch: Chromosome, instance: ProblemInstance) -> Schedule:
    """
    Decoder-oriented feasible scheduling approximation.
    It follows the documented 'priority-based recursive scheduling with robust buffers' design.
    """
    tasks = instance.tasks
    seq = sequence_by_device(ch, instance)

    # Device states
    train_time = [0.0] * instance.equipment.n_train_rs
    train_pos = [(0.0, 0.0)] * instance.equipment.n_train_rs
    train_idle_cost = 0.0
    train_move_cost = 0.0
    train_op_cost = 0.0

    yard_time = [0.0] * instance.equipment.n_yard_rs
    yard_pos = [instance.yard_zones_xy[1]] * instance.equipment.n_yard_rs
    yard_idle_cost = 0.0
    yard_move_cost = 0.0
    yard_op_cost = 0.0

    truck_time = [0.0] * total_trucks(instance)
    truck_pos = [instance.gate_xy] * total_trucks(instance)
    truck_idle_cost = 0.0
    truck_move_cost = 0.0

    train_count_on_device = [0] * instance.equipment.n_train_rs
    yard_count_on_device = [0] * instance.equipment.n_yard_rs

    records: Dict[int, TaskScheduleRecord] = {}
    violation = 0.0
    feasible = True

    # Decode globally by priority, while each task respects its assigned devices.
    for tid in ch.priority:
        task = tasks[tid]
        r = ch.train_assign[tid]
        k = ch.truck_assign[tid]
        y = ch.yard_assign[tid]

        rec = TaskScheduleRecord(task_id=tid, train_rs_id=r, truck_id=k, yard_rs_id=(y if y >= 0 else None))

        # --- train reach stacker movement to rail point
        train_move_dist = manhattan_distance(train_pos[r], task.rail_point_xy)
        train_move_time = km_to_hours(train_move_dist, instance.speed.train_rs_speed_kmph)
        train_ready = train_time[r] + train_move_time

        # robust buffer on first gamma_1 tasks of each train/yard device
        train_buffer = robust_buffer_for_train_task(task) if train_count_on_device[r] < instance.robust.gamma_1 else 0.0

        # --- truck movement logic by task type
        loaded_dist = task_loaded_distance(task, instance)
        truck_loaded_time = km_to_hours(loaded_dist, instance.speed.truck_loaded_speed_kmph)

        if task.task_type in {"A", "E"}:
            truck_to_line_dist = manhattan_distance(truck_pos[k], task.rail_point_xy)
        elif task.task_type == "B":
            truck_to_line_dist = manhattan_distance(truck_pos[k], task.rail_point_xy)
        elif task.task_type == "C":
            truck_to_line_dist = manhattan_distance(truck_pos[k], task.yard_zone_xy)
        elif task.task_type == "D":
            truck_to_line_dist = manhattan_distance(truck_pos[k], task.rail_point_xy)
        else:
            raise ValueError(task.task_type)

        truck_to_line_time = km_to_hours(truck_to_line_dist, instance.speed.truck_empty_speed_kmph)
        truck_ready = truck_time[k] + truck_to_line_time

        # external truck elastic time window
        if task.task_type in {"B", "D"} and task.arrival_window is not None:
            earliest_allowed = task.arrival_window.start - instance.robust.ext_early_slack
            latest_allowed = task.arrival_window.end + instance.robust.ext_late_slack
            if truck_ready < earliest_allowed:
                rec.early = earliest_allowed - truck_ready
            if truck_ready > latest_allowed:
                feasible = False
                violation += truck_ready - latest_allowed
                rec.feasible = False
                rec.notes.append("external truck arrived outside elastic window")

        # --- process-specific scheduling
        if task.task_type in {"A", "B", "E"}:
            # line service first
            line_start = max(train_ready, truck_ready)
            rec.truck_line_arrive = truck_ready
            rec.train_start = line_start
            rec.train_finish = line_start + task.train_service_time + train_buffer
            rec.truck_line_leave = rec.train_finish

            train_idle_gap = max(0.0, line_start - train_ready)
            train_idle_cost += min(train_idle_gap, instance.decoder.t1_idle_shutdown_threshold) * \
                instance.energy.train_idle_power * instance.energy.electricity_price
            truck_idle_gap = max(0.0, line_start - truck_ready)
            truck_idle_cost += min(truck_idle_gap, instance.decoder.t2_truck_shutdown_threshold) * \
                instance.energy.truck_idle_fuel_per_hour * instance.energy.fuel_price

            train_move_cost += train_move_time * instance.energy.train_move_power * instance.energy.electricity_price
            train_op_cost += task.train_service_time * instance.energy.train_op_power * instance.energy.electricity_price

            # follow-up
            if task.task_type in {"A", "E"}:
                # to yard, then yard service
                if y is None or y < 0:
                    feasible = False
                    violation += 1.0
                    rec.feasible = False
                    rec.notes.append("yard task without yard equipment")
                else:
                    yard_move_dist = manhattan_distance(yard_pos[y], task.yard_zone_xy)
                    yard_move_time = km_to_hours(yard_move_dist, instance.speed.yard_rs_speed_kmph)
                    yard_ready = yard_time[y] + yard_move_time
                    yard_buffer = robust_buffer_for_yard_task(task) if yard_count_on_device[y] < instance.robust.gamma_1 else 0.0

                    truck_arrive_yard = rec.truck_line_leave + truck_loaded_time
                    yard_start = max(yard_ready, truck_arrive_yard)

                    rec.yard_arrive = truck_arrive_yard
                    rec.yard_finish = yard_start + task.yard_service_time + yard_buffer

                    yard_idle_gap = max(0.0, yard_start - yard_ready)
                    yard_idle_cost += min(yard_idle_gap, instance.decoder.t1_idle_shutdown_threshold) * \
                        instance.energy.yard_idle_power * instance.energy.electricity_price
                    truck_idle_gap_2 = max(0.0, yard_start - truck_arrive_yard)
                    truck_idle_cost += min(truck_idle_gap_2, instance.decoder.t2_truck_shutdown_threshold) * \
                        instance.energy.truck_idle_fuel_per_hour * instance.energy.fuel_price

                    yard_move_cost += yard_move_time * instance.energy.yard_move_power * instance.energy.electricity_price
                    yard_op_cost += task.yard_service_time * instance.energy.yard_op_power * instance.energy.electricity_price

                    yard_time[y] = rec.yard_finish
                    yard_pos[y] = task.yard_zone_xy
                    yard_count_on_device[y] += 1
                    truck_time[k] = rec.yard_finish
                    truck_pos[k] = task.yard_zone_xy
            else:
                # B goes to gate and finishes
                truck_time[k] = rec.truck_line_leave + truck_loaded_time
                truck_pos[k] = instance.gate_xy

        elif task.task_type == "C":
            # yard first, then line
            if y is None or y < 0:
                feasible = False
                violation += 1.0
                rec.feasible = False
                rec.notes.append("export-from-yard task without yard equipment")
            else:
                yard_move_dist = manhattan_distance(yard_pos[y], task.yard_zone_xy)
                yard_move_time = km_to_hours(yard_move_dist, instance.speed.yard_rs_speed_kmph)
                yard_ready = yard_time[y] + yard_move_time
                yard_buffer = robust_buffer_for_yard_task(task) if yard_count_on_device[y] < instance.robust.gamma_1 else 0.0

                truck_to_yard_dist = manhattan_distance(truck_pos[k], task.yard_zone_xy)
                truck_to_yard_time = km_to_hours(truck_to_yard_dist, instance.speed.truck_empty_speed_kmph)
                truck_ready_yard = truck_time[k] + truck_to_yard_time

                yard_start = max(yard_ready, truck_ready_yard)
                yard_finish = yard_start + task.yard_service_time + yard_buffer

                rec.yard_arrive = truck_ready_yard
                rec.yard_finish = yard_finish

                yard_idle_gap = max(0.0, yard_start - yard_ready)
                yard_idle_cost += min(yard_idle_gap, instance.decoder.t1_idle_shutdown_threshold) * \
                    instance.energy.yard_idle_power * instance.energy.electricity_price
                truck_idle_gap = max(0.0, yard_start - truck_ready_yard)
                truck_idle_cost += min(truck_idle_gap, instance.decoder.t2_truck_shutdown_threshold) * \
                    instance.energy.truck_idle_fuel_per_hour * instance.energy.fuel_price

                yard_move_cost += yard_move_time * instance.energy.yard_move_power * instance.energy.electricity_price
                yard_op_cost += task.yard_service_time * instance.energy.yard_op_power * instance.energy.electricity_price

                # then line
                truck_arrive_line = yard_finish + truck_loaded_time
                line_start = max(train_ready, truck_arrive_line)
                rec.truck_line_arrive = truck_arrive_line
                rec.train_start = line_start
                rec.train_finish = line_start + task.train_service_time + train_buffer
                rec.truck_line_leave = rec.train_finish

                train_idle_gap = max(0.0, line_start - train_ready)
                train_idle_cost += min(train_idle_gap, instance.decoder.t1_idle_shutdown_threshold) * \
                    instance.energy.train_idle_power * instance.energy.electricity_price
                truck_idle_gap_2 = max(0.0, line_start - truck_arrive_line)
                truck_idle_cost += min(truck_idle_gap_2, instance.decoder.t2_truck_shutdown_threshold) * \
                    instance.energy.truck_idle_fuel_per_hour * instance.energy.fuel_price

                train_move_cost += train_move_time * instance.energy.train_move_power * instance.energy.electricity_price
                train_op_cost += task.train_service_time * instance.energy.train_op_power * instance.energy.electricity_price

                yard_time[y] = yard_finish
                yard_pos[y] = task.yard_zone_xy
                yard_count_on_device[y] += 1

                truck_time[k] = rec.truck_line_leave
                truck_pos[k] = task.rail_point_xy

        elif task.task_type == "D":
            # external truck arrives directly to line and loaded onto train
            line_start = max(train_ready, truck_ready)
            rec.truck_line_arrive = truck_ready
            rec.train_start = line_start
            rec.train_finish = line_start + task.train_service_time + train_buffer
            rec.truck_line_leave = rec.train_finish

            if task.arrival_window is not None:
                rec.early = max(0.0, task.arrival_window.start - truck_ready)
                rec.late = max(0.0, truck_ready - task.arrival_window.end)

            train_idle_gap = max(0.0, line_start - train_ready)
            train_idle_cost += min(train_idle_gap, instance.decoder.t1_idle_shutdown_threshold) * \
                instance.energy.train_idle_power * instance.energy.electricity_price
            truck_idle_gap = max(0.0, line_start - truck_ready)
            truck_idle_cost += min(truck_idle_gap, instance.decoder.t2_truck_shutdown_threshold) * \
                instance.energy.truck_idle_fuel_per_hour * instance.energy.fuel_price

            train_move_cost += train_move_time * instance.energy.train_move_power * instance.energy.electricity_price
            train_op_cost += task.train_service_time * instance.energy.train_op_power * instance.energy.electricity_price

            truck_time[k] = rec.truck_line_leave
            truck_pos[k] = task.rail_point_xy

        # Time window penalty
        if task.task_type in {"B", "D"} and task.arrival_window is not None:
            if rec.truck_line_arrive < task.arrival_window.start:
                rec.early = task.arrival_window.start - rec.truck_line_arrive
            if rec.truck_line_arrive > task.arrival_window.end:
                rec.late = rec.truck_line_arrive - task.arrival_window.end

        # Update train device state
        train_time[r] = rec.train_finish
        train_pos[r] = task.rail_point_xy
        train_count_on_device[r] += 1

        # Update truck movement energy
        truck_move_cost += truck_to_line_dist * instance.energy.truck_empty_fuel_per_km * instance.energy.fuel_price
        truck_move_cost += loaded_dist * instance.energy.truck_loaded_fuel_per_km * instance.energy.fuel_price

        records[tid] = rec

    makespan = 0.0
    for rec in records.values():
        if rec.yard_finish is not None:
            makespan = max(makespan, rec.yard_finish)
        makespan = max(makespan, rec.train_finish)

    tw_penalty = 0.0
    for tid, rec in records.items():
        task = tasks[tid]
        if task.task_type in {"B", "D"}:
            tw_penalty += rec.early * instance.energy.early_penalty_per_hour
            tw_penalty += rec.late * instance.energy.late_penalty_per_hour

    baseline_energy_cost = train_op_cost + train_move_cost + train_idle_cost + \
                           yard_op_cost + yard_move_cost + yard_idle_cost + \
                           truck_move_cost + truck_idle_cost

    objective_2_baseline = baseline_energy_cost + tw_penalty

    # Device start/stop frequency rough check
    # Each short/long idle split can imply one start-stop. Here we use number of tasks as a rough upper bound.
    for r, count in enumerate(train_count_on_device):
        if count > instance.decoder.nmax_start_stop + 1:
            feasible = False
            violation += (count - instance.decoder.nmax_start_stop - 1)
    for y, count in enumerate(yard_count_on_device):
        if count > instance.decoder.nmax_start_stop + 1:
            feasible = False
            violation += (count - instance.decoder.nmax_start_stop - 1)

    return Schedule(
        task_records=records,
        makespan=makespan,
        baseline_energy_cost=baseline_energy_cost,
        time_window_penalty=tw_penalty,
        objective_2_baseline=objective_2_baseline,
        feasible=feasible,
        constraint_violation=violation,
        device_sequences=seq,
    )


# ============================================================
# 7. Failure simulation and CVaR
# ============================================================

def generate_failure_events(instance: ProblemInstance, horizon: float, seed: int) -> List[FailureEvent]:
    rng = random.Random(seed)
    events: List[FailureEvent] = []

    def sample_events(kind: str, n_devices: int, rate: float, mean_repair: float) -> None:
        for device_id in range(n_devices):
            t = 0.0
            while t < horizon:
                if rate <= 0:
                    break
                delta = rng.expovariate(rate)
                t += delta
                if t >= horizon:
                    break
                duration = max(0.01, rng.expovariate(1.0 / mean_repair))
                events.append(FailureEvent(kind, device_id, t, duration))
                t += duration

    sample_events("train_rs", instance.equipment.n_train_rs, instance.failure.lambda_train_rs, instance.failure.repair_time_mean_train_rs)
    sample_events("yard_rs", instance.equipment.n_yard_rs, instance.failure.lambda_yard_rs, instance.failure.repair_time_mean_yard_rs)
    sample_events("truck", total_trucks(instance), instance.failure.lambda_truck, instance.failure.repair_time_mean_truck)

    events.sort(key=lambda e: e.start_time)
    return events


def record_finish_time(rec: TaskScheduleRecord) -> float:
    if rec.yard_finish is not None:
        return rec.yard_finish
    return rec.train_finish


def estimate_task_delay_due_to_failure(rec: TaskScheduleRecord, event: FailureEvent) -> float:
    # If device failure overlaps task execution or before task start on same device.
    overlap = 0.0
    if event.device_kind == "train_rs" and rec.train_rs_id == event.device_id:
        if rec.train_start <= event.start_time <= rec.train_finish:
            overlap = event.duration
        elif event.start_time <= rec.train_start:
            overlap = event.duration * 0.5
    elif event.device_kind == "yard_rs" and rec.yard_rs_id == event.device_id and rec.yard_finish is not None:
        yard_start_est = rec.yard_finish  # conservative fallback since exact yard start not stored
        if event.start_time <= yard_start_est:
            overlap = event.duration * 0.5
    elif event.device_kind == "truck" and rec.truck_id == event.device_id:
        if rec.truck_line_arrive <= event.start_time <= rec.truck_line_leave:
            overlap = event.duration
        elif event.start_time <= rec.truck_line_arrive:
            overlap = event.duration * 0.5
    return overlap


def fast_reschedule_failure_cost(schedule: Schedule, instance: ProblemInstance, events: List[FailureEvent]) -> float:
    """
    Engineering approximation of the document's
    'earliest available - minimum incremental energy' fast rescheduling rule.
    """
    extra_idle = 0.0
    extra_penalty = 0.0
    extra_move = 0.0

    # Determine impacted tasks and delay them according to failures
    delayed_finish: Dict[int, float] = {tid: record_finish_time(rec) for tid, rec in schedule.task_records.items()}

    for event in events:
        extra_idle += event.duration * instance.energy.failure_idle_energy_cost_per_hour

        impacted = []
        for tid, rec in schedule.task_records.items():
            delay = estimate_task_delay_due_to_failure(rec, event)
            if delay > 0:
                impacted.append((tid, delay))

        impacted.sort(key=lambda x: schedule.task_records[x[0]].train_start)
        for tid, delay in impacted:
            delayed_finish[tid] += delay
            # approximate incremental movement/coordination energy
            extra_move += instance.failure.gamma_reschedule * delay * (
                instance.energy.train_move_power * instance.energy.electricity_price
            )

            task = instance.tasks[tid]
            if task.task_type in {"B", "D"} and task.arrival_window is not None:
                rec = schedule.task_records[tid]
                worsened_late = max(0.0, (rec.late + delay) - rec.late)
                extra_penalty += worsened_late * instance.energy.late_penalty_per_hour

    return extra_idle + extra_move + extra_penalty


def compute_cvar(losses: List[float], alpha: float) -> float:
    if not losses:
        return 0.0
    var_alpha = percentile(losses, alpha)
    tail = [x for x in losses if x >= var_alpha]
    if not tail:
        return var_alpha
    return sum(tail) / len(tail)


def evaluate_failure_risk(schedule: Schedule, instance: ProblemInstance) -> float:
    q = instance.failure.monte_carlo_scenarios
    horizon = max(schedule.makespan, 0.1)
    losses = []

    for s in range(q):
        seed = instance.failure.common_random_seed + s
        events = generate_failure_events(instance, horizon, seed)
        loss = fast_reschedule_failure_cost(schedule, instance, events)
        losses.append(loss)

    return compute_cvar(losses, instance.robust.alpha)


# ============================================================
# 8. Evaluation
# ============================================================

def evaluate_chromosome(ch: Chromosome, instance: ProblemInstance) -> EvaluationResult:
    schedule = decode_chromosome(ch, instance)
    failure_cvar_cost = evaluate_failure_risk(schedule, instance)
    f2 = schedule.objective_2_baseline + instance.robust.beta * failure_cvar_cost
    return EvaluationResult(
        chromosome=ch,
        feasible=schedule.feasible,
        constraint_violation=schedule.constraint_violation,
        f1_makespan=schedule.makespan,
        f2_cost=f2,
        baseline_cost=schedule.objective_2_baseline,
        failure_cvar_cost=failure_cvar_cost,
        schedule=schedule,
    )


# ============================================================
# 9. NSGA-II core
# ============================================================

def constraint_dominates(a: EvaluationResult, b: EvaluationResult) -> bool:
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.constraint_violation < b.constraint_violation
    better_or_equal = a.f1_makespan <= b.f1_makespan and a.f2_cost <= b.f2_cost
    strictly_better = a.f1_makespan < b.f1_makespan or a.f2_cost < b.f2_cost
    return better_or_equal and strictly_better


def fast_non_dominated_sort(pop: List[EvaluationResult]) -> List[List[int]]:
    S = [[] for _ in pop]
    n = [0 for _ in pop]
    fronts: List[List[int]] = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if constraint_dominates(pop[p], pop[q]):
                S[p].append(q)
            elif constraint_dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def crowding_distance(front: List[EvaluationResult]) -> List[float]:
    if not front:
        return []
    n = len(front)
    if n <= 2:
        return [float("inf")] * n

    d = [0.0] * n
    for key in ["f1_makespan", "f2_cost"]:
        values = [getattr(ind, key) for ind in front]
        lo, hi = min(values), max(values)
        order = sorted(range(n), key=lambda i: values[i])
        d[order[0]] = d[order[-1]] = float("inf")
        if hi == lo:
            continue
        for j in range(1, n - 1):
            prev_v = values[order[j - 1]]
            next_v = values[order[j + 1]]
            d[order[j]] += (next_v - prev_v) / (hi - lo)
    return d


def tournament_select(pop: List[EvaluationResult], ranks: Dict[int, int], crowd: Dict[int, float], rng: random.Random) -> EvaluationResult:
    candidates = rng.sample(range(len(pop)), 2)
    a, b = candidates[0], candidates[1]
    if ranks[a] < ranks[b]:
        return pop[a]
    if ranks[b] < ranks[a]:
        return pop[b]
    return pop[a] if crowd[a] >= crowd[b] else pop[b]


def uniform_crossover(a: List[int], b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    c1, c2 = [], []
    for x, y in zip(a, b):
        if rng.random() < 0.5:
            c1.append(x); c2.append(y)
        else:
            c1.append(y); c2.append(x)
    return c1, c2


def order_crossover(p1: List[int], p2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    n = len(p1)
    i, j = sorted(rng.sample(range(n), 2))
    child1 = [-1] * n
    child2 = [-1] * n
    child1[i:j+1] = p1[i:j+1]
    child2[i:j+1] = p2[i:j+1]

    def fill(child: List[int], donor: List[int]) -> None:
        pos = (j + 1) % n
        donor_pos = (j + 1) % n
        used = set(child[i:j+1])
        while -1 in child:
            gene = donor[donor_pos]
            if gene not in used:
                child[pos] = gene
                used.add(gene)
                pos = (pos + 1) % n
            donor_pos = (donor_pos + 1) % n

    fill(child1, p2)
    fill(child2, p1)
    return child1, child2


def crossover(parent1: Chromosome, parent2: Chromosome, instance: ProblemInstance, ga: GAParams, rng: random.Random) -> Tuple[Chromosome, Chromosome]:
    if rng.random() > ga.crossover_prob:
        return parent1.clone(), parent2.clone()

    train1, train2 = uniform_crossover(parent1.train_assign, parent2.train_assign, rng)
    truck1, truck2 = uniform_crossover(parent1.truck_assign, parent2.truck_assign, rng)
    yard1, yard2 = uniform_crossover(parent1.yard_assign, parent2.yard_assign, rng)
    prio1, prio2 = order_crossover(parent1.priority, parent2.priority, rng)

    c1 = repair_chromosome(Chromosome(train1, truck1, yard1, prio1), instance, rng)
    c2 = repair_chromosome(Chromosome(train2, truck2, yard2, prio2), instance, rng)
    return c1, c2


def mutate(ch: Chromosome, instance: ProblemInstance, ga: GAParams, rng: random.Random) -> Chromosome:
    n = len(instance.tasks)

    for i, task in enumerate(instance.tasks):
        if rng.random() < ga.mutation_prob:
            ch.train_assign[i] = rng.randrange(instance.equipment.n_train_rs)
        if rng.random() < ga.mutation_prob:
            ch.truck_assign[i] = rng.choice(legal_truck_ids_for_task(task, instance))
        if rng.random() < ga.mutation_prob:
            ch.yard_assign[i] = rng.choice(legal_yard_ids_for_task(task, instance))

    if rng.random() < ga.mutation_prob:
        i, j = sorted(rng.sample(range(n), 2))
        if rng.random() < 0.5:
            ch.priority[i], ch.priority[j] = ch.priority[j], ch.priority[i]
        else:
            ch.priority[i:j+1] = list(reversed(ch.priority[i:j+1]))

    return repair_chromosome(ch, instance, rng)


def build_archive(pop: List[EvaluationResult], archive_max_size: int) -> List[EvaluationResult]:
    fronts = fast_non_dominated_sort(pop)
    archive: List[EvaluationResult] = []
    for front_idx in fronts:
        front = [pop[i] for i in front_idx]
        if len(archive) + len(front) <= archive_max_size:
            archive.extend(front)
        else:
            crowd = crowding_distance(front)
            ranked = sorted(zip(front, crowd), key=lambda x: x[1], reverse=True)
            archive.extend([ind for ind, _ in ranked[:max(0, archive_max_size - len(archive))]])
            break
    return archive


def nsga2_optimize(instance: ProblemInstance, ga: GAParams) -> Dict[str, object]:
    rng = random.Random(ga.seed)
    population = initialize_population(instance, ga, rng)
    eval_pop = [evaluate_chromosome(ind, instance) for ind in population]

    archive = build_archive(eval_pop, ga.archive_max_size)
    history = []
    best_front_signature = None
    stagnation = 0

    for gen in range(ga.n_generations):
        fronts = fast_non_dominated_sort(eval_pop)
        ranks = {}
        crowd_map = {}

        for rank, front_idx in enumerate(fronts):
            front = [eval_pop[i] for i in front_idx]
            crowd = crowding_distance(front)
            for local_idx, global_idx in enumerate(front_idx):
                ranks[global_idx] = rank
                crowd_map[global_idx] = crowd[local_idx]

        offspring: List[Chromosome] = []
        while len(offspring) < ga.population_size:
            p1 = tournament_select(eval_pop, ranks, crowd_map, rng).chromosome.clone()
            p2 = tournament_select(eval_pop, ranks, crowd_map, rng).chromosome.clone()
            c1, c2 = crossover(p1, p2, instance, ga, rng)
            c1 = mutate(c1, instance, ga, rng)
            c2 = mutate(c2, instance, ga, rng)
            offspring.append(c1)
            if len(offspring) < ga.population_size:
                offspring.append(c2)

        eval_off = [evaluate_chromosome(ind, instance) for ind in offspring]
        merged = eval_pop + eval_off
        merged_fronts = fast_non_dominated_sort(merged)

        next_pop: List[EvaluationResult] = []
        for front_idx in merged_fronts:
            front = [merged[i] for i in front_idx]
            if len(next_pop) + len(front) <= ga.population_size:
                next_pop.extend(front)
            else:
                crowd = crowding_distance(front)
                ranked = sorted(zip(front, crowd), key=lambda x: x[1], reverse=True)
                need = ga.population_size - len(next_pop)
                next_pop.extend([ind for ind, _ in ranked[:need]])
                break

        eval_pop = next_pop
        archive = build_archive(archive + eval_pop, ga.archive_max_size)

        current_front = fast_non_dominated_sort(eval_pop)[0]
        signature = [(round(eval_pop[i].f1_makespan, 4), round(eval_pop[i].f2_cost, 4)) for i in current_front]
        signature = tuple(sorted(signature))
        if signature == best_front_signature:
            stagnation += 1
        else:
            stagnation = 0
            best_front_signature = signature

        history.append({
            "generation": gen,
            "n_archive": len(archive),
            "best_front_size": len(current_front),
            "best_f1": min(ind.f1_makespan for ind in eval_pop),
            "best_f2": min(ind.f2_cost for ind in eval_pop),
        })

        if stagnation >= ga.max_no_improve_generations:
            break

    final_front_indices = fast_non_dominated_sort(archive)[0] if archive else []
    pareto = [archive[i] for i in final_front_indices]
    pareto.sort(key=lambda x: (x.f1_makespan, x.f2_cost))
    return {
        "pareto": pareto,
        "archive": archive,
        "history": history,
        "final_population": eval_pop,
    }


# ============================================================
# 10. Experiment metrics
# ============================================================

def spacing_metric(front: List[EvaluationResult]) -> float:
    if len(front) <= 2:
        return 1.0
    pts = [(x.f1_makespan, x.f2_cost) for x in front]
    dists = []
    for i, p in enumerate(pts):
        nearest = min(
            math.dist(p, q) for j, q in enumerate(pts) if j != i
        )
        dists.append(nearest)
    mean_d = sum(dists) / len(dists)
    if mean_d == 0:
        return 1.0
    denom = len(dists) * mean_d
    return max(0.0, 1.0 - math.sqrt(sum((d - mean_d) ** 2 for d in dists) / denom))


def gd_metric(front: List[EvaluationResult], ref_front: List[Tuple[float, float]]) -> float:
    if not front or not ref_front:
        return float("nan")
    s = 0.0
    for x in front:
        d = min(math.dist((x.f1_makespan, x.f2_cost), r) for r in ref_front)
        s += d * d
    return math.sqrt(s / len(front))


def igd_metric(front: List[EvaluationResult], ref_front: List[Tuple[float, float]]) -> float:
    if not front or not ref_front:
        return float("nan")
    pts = [(x.f1_makespan, x.f2_cost) for x in front]
    s = 0.0
    for r in ref_front:
        d = min(math.dist(r, p) for p in pts)
        s += d
    return s / len(ref_front)


# ============================================================
# 11. Baselines and experiment runners
# ============================================================

def deterministic_cost_of_chromosome(ch: Chromosome, instance: ProblemInstance) -> EvaluationResult:
    schedule = decode_chromosome(ch, instance)
    return EvaluationResult(
        chromosome=ch,
        feasible=schedule.feasible,
        constraint_violation=schedule.constraint_violation,
        f1_makespan=schedule.makespan,
        f2_cost=schedule.objective_2_baseline,
        baseline_cost=schedule.objective_2_baseline,
        failure_cvar_cost=0.0,
        schedule=schedule,
    )


def run_baseline_m0_nsga(instance: ProblemInstance, ga: GAParams) -> Dict[str, object]:
    rng = random.Random(ga.seed)
    population = initialize_population(instance, ga, rng)
    eval_pop = [deterministic_cost_of_chromosome(ind, instance) for ind in population]
    archive = build_archive(eval_pop, ga.archive_max_size)
    history = []

    for gen in range(ga.n_generations):
        fronts = fast_non_dominated_sort(eval_pop)
        ranks, crowd_map = {}, {}
        for rank, front_idx in enumerate(fronts):
            front = [eval_pop[i] for i in front_idx]
            crowd = crowding_distance(front)
            for local, global_idx in enumerate(front_idx):
                ranks[global_idx] = rank
                crowd_map[global_idx] = crowd[local]

        offspring = []
        while len(offspring) < ga.population_size:
            p1 = tournament_select(eval_pop, ranks, crowd_map, rng).chromosome.clone()
            p2 = tournament_select(eval_pop, ranks, crowd_map, rng).chromosome.clone()
            c1, c2 = crossover(p1, p2, instance, ga, rng)
            offspring.append(mutate(c1, instance, ga, rng))
            if len(offspring) < ga.population_size:
                offspring.append(mutate(c2, instance, ga, rng))

        eval_off = [deterministic_cost_of_chromosome(ind, instance) for ind in offspring]
        merged = eval_pop + eval_off
        merged_fronts = fast_non_dominated_sort(merged)
        next_pop = []
        for front_idx in merged_fronts:
            front = [merged[i] for i in front_idx]
            if len(next_pop) + len(front) <= ga.population_size:
                next_pop.extend(front)
            else:
                crowd = crowding_distance(front)
                ranked = sorted(zip(front, crowd), key=lambda x: x[1], reverse=True)
                next_pop.extend([ind for ind, _ in ranked[:ga.population_size - len(next_pop)]])
                break
        eval_pop = next_pop
        archive = build_archive(archive + eval_pop, ga.archive_max_size)
        history.append({"generation": gen, "archive_size": len(archive)})

    final_front = [archive[i] for i in fast_non_dominated_sort(archive)[0]] if archive else []
    return {"pareto": final_front, "archive": archive, "history": history}


def run_heuristic_rule(instance: ProblemInstance, rule: str) -> List[EvaluationResult]:
    rng = random.Random(123)
    ch = make_heuristic_individual(instance, rule, rng)
    return [evaluate_chromosome(ch, instance)]


def parameter_sensitivity_experiment(
    instance: ProblemInstance,
    ga: GAParams,
    gamma_values: Iterable[int],
    beta_values: Iterable[float],
) -> List[Dict[str, float]]:
    rows = []
    for gamma_1 in gamma_values:
        for beta in beta_values:
            inst = copy.deepcopy(instance)
            inst.robust.gamma_1 = gamma_1
            inst.robust.beta = beta
            result = nsga2_optimize(inst, ga)
            pareto = result["pareto"]
            if not pareto:
                rows.append({"gamma_1": gamma_1, "beta": beta, "best_f1": math.nan, "best_f2": math.nan})
                continue
            best_f1 = min(x.f1_makespan for x in pareto)
            best_f2 = min(x.f2_cost for x in pareto)
            rows.append({"gamma_1": gamma_1, "beta": beta, "best_f1": best_f1, "best_f2": best_f2})
    return rows


# ============================================================
# 12. Output helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pareto_csv(front: List[EvaluationResult], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "makespan", "total_cost", "baseline_cost", "failure_cvar_cost",
            "feasible", "constraint_violation"
        ])
        for ind in front:
            writer.writerow([
                ind.f1_makespan, ind.f2_cost, ind.baseline_cost, ind.failure_cvar_cost,
                int(ind.feasible), ind.constraint_violation
            ])


def save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_json(data: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 13. Main demo / experiment entry
# ============================================================

def run_full_demo(output_dir: str = "outputs", scale: str = "small") -> Dict[str, object]:
    instance = build_default_instance(scale=scale, seed=7)
    ga = GAParams(
        population_size=default_population_size(len(instance.tasks)),
        n_generations=40 if scale == "small" else 30,
        archive_max_size=200,
        seed=42,
    )

    out = Path(output_dir)
    ensure_dir(out)

    t0 = time.perf_counter()
    sn_result = nsga2_optimize(instance, ga)
    t_sn = time.perf_counter() - t0

    t1 = time.perf_counter()
    m0_result = run_baseline_m0_nsga(instance, ga)
    t_m0 = time.perf_counter() - t1

    heur_edd = run_heuristic_rule(instance, "EDD")
    heur_spt = run_heuristic_rule(instance, "SPT")

    save_pareto_csv(sn_result["pareto"], out / f"pareto_sn_nsga2_{scale}.csv")
    save_pareto_csv(m0_result["pareto"], out / f"pareto_m0_nsga2_{scale}.csv")
    save_history_csv(sn_result["history"], out / f"history_sn_nsga2_{scale}.csv")
    save_history_csv(m0_result["history"], out / f"history_m0_nsga2_{scale}.csv")

    sensitivity = parameter_sensitivity_experiment(
        instance,
        ga,
        gamma_values=[0, 1, 3, 5],
        beta_values=[0.0, 0.5, 1.0, 2.0]
    )
    ensure_dir(out)
    with (out / f"sensitivity_{scale}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity)

    summary = {
        "instance": instance.name,
        "scale": scale,
        "n_tasks": len(instance.tasks),
        "sn_nsga2_runtime_sec": t_sn,
        "m0_nsga2_runtime_sec": t_m0,
        "sn_pareto_size": len(sn_result["pareto"]),
        "m0_pareto_size": len(m0_result["pareto"]),
        "best_sn_f1": min((x.f1_makespan for x in sn_result["pareto"]), default=None),
        "best_sn_f2": min((x.f2_cost for x in sn_result["pareto"]), default=None),
        "best_m0_f1": min((x.f1_makespan for x in m0_result["pareto"]), default=None),
        "best_m0_f2": min((x.f2_cost for x in m0_result["pareto"]), default=None),
        "heuristic_edd": {
            "f1": heur_edd[0].f1_makespan,
            "f2": heur_edd[0].f2_cost,
        },
        "heuristic_spt": {
            "f1": heur_spt[0].f1_makespan,
            "f2": heur_spt[0].f2_cost,
        },
    }
    save_json(summary, out / f"summary_{scale}.json")
    return summary


if __name__ == "__main__":
    summary = run_full_demo(output_dir="outputs", scale="small")
    print("Demo finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
'''

readme = r'''# SN-NSGA-II 实验代码说明

本代码根据你上传的两份文档整理生成，结构对应论文中的：
- 第三章：陆港正面吊-集卡协同调度混合鲁棒优化模型
- 第四章：仿真嵌套的改进非支配排序遗传算法（SN-NSGA-II）
- 第五章：算例实验与敏感性分析

## 代码功能
- 三段式编码：
  - 第一段：装卸线正面吊分配
  - 第二段：堆场正面吊与集卡分配
  - 第三段：任务优先级排序
- 解码排程
- 第一层鲁棒：作业时间缓冲（Gamma_1）
- 第二