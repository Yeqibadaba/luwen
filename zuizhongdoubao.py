import numpy as np
import random
import copy
from typing import List, Tuple, Dict

# ====================== 1. 可调参数设置  ======================
# 算法参数
POPULATION_SIZE = 100  # 种群规模
MAX_ITERATIONS = 100  # 最大迭代次数
CROSSOVER_PROB = 0.85  # 交叉概率
MUTATION_PROB = 0.1  # 变异概率
SIMULATION_SCENARIOS = 100  # 蒙特卡洛仿真情景数（对应第三章3.4.3）
RISK_AVERSION_COEF = 1.5  # 风险厌恶系数 β（对应第三章3.5.1）
GAM MA = 0.1  # 重调度能耗-时间平衡系数 γ（对应第三章3.4.3）
ROBUST_BUDGET = 5  # 第一层鲁棒预算 Γ1（对应第三章3.4.1）

# 问题实例参数
NUM_TASKS = 40  # 任务总数
NUM_RTG = 3  # 火车线正面吊数量
NUM_YARD = 3  # 堆场正面吊数量
NUM_TRUCKS = 8  # 集卡数量
TASK_TYPES = ['A', 'B', 'C', 'D', 'E']  # 对应第三章3.1.2的五种任务类型

# 设备参数（对应第三章3.1.3）
RTG_SPEED = 1.0  # 火车正面吊移动速度
YARD_SPEED = 1.2  # 堆场正面吊移动速度
TRUCK_SPEED = 2.5  # 集卡移动速度
RTG_IDLE_ENERGY = 0.5  # 火车正面吊单位时间闲置能耗
YARD_IDLE_ENERGY = 0.4  # 堆场正面吊单位时间闲置能耗
TRUCK_IDLE_ENERGY = 0.3  # 集卡单位时间闲置能耗
TRUCK_UNIT_DIST_ENERGY = 0.2  # 集卡单位距离能耗

# 故障参数（对应第三章3.4.3）
RTG_FAILURE_RATE = 0.008  # 火车正面吊单位时间故障率 λ
YARD_FAILURE_RATE = 0.006  # 堆场正面吊单位时间故障率
TRUCK_FAILURE_RATE = 0.01  # 集卡单位时间故障率
REPAIR_TIME_MEAN = 4.0  # 平均维修时间
REPAIR_TIME_STD = 1.2  # 维修时间标准差


# ====================== 2. 问题实例数据生成（模拟真实陆港数据） ======================
def generate_problem_instance() -> Dict:
    """生成符合第三章3.1节描述的陆港问题实例"""
    tasks = []
    for i in range(NUM_TASKS):
        task_type = random.choice(TASK_TYPES)
        # 生成符合第三章3.2节符号定义的参数
        rtg_time = random.uniform(1.5, 4.5)  # p^rtg_i：火车正面吊标准作业时间
        yard_time = random.uniform(1.5, 4.0) if task_type in ['A', 'C', 'E'] else 0.0  # p^yard_i
        # 外部集卡任务（B/D）有时间窗，其他无
        time_window_start = random.uniform(0, 80) if task_type in ['B', 'D'] else 0.0
        time_window_end = time_window_start + random.uniform(12, 25) if task_type in ['B', 'D'] else 10000.0
        delta_time = random.uniform(3, 8) if task_type in ['B', 'D'] else 0.0  # Δ_i：到达扰动容限
        # 简化移动距离（实际可按第三章3.1.3曼哈顿距离计算）
        dist_rtg_yard = random.uniform(4, 14) if task_type in ['A', 'C', 'E'] else 0.0
        dist_rtg_gate = random.uniform(8, 18) if task_type in ['B', 'D'] else 0.0
        # 作业时间最大偏差（对应第三章3.4.1不确定集）
        rtg_time_dev = rtg_time * 0.2
        yard_time_dev = yard_time * 0.2 if task_type in ['A', 'C', 'E'] else 0.0

        tasks.append({
            'id': i,
            'type': task_type,
            'rtg_time': rtg_time,
            'rtg_time_dev': rtg_time_dev,
            'yard_time': yard_time,
            'yard_time_dev': yard_time_dev,
            'time_window': [time_window_start, time_window_end],
            'delta_time': delta_time,
            'dist_rtg_yard': dist_rtg_yard,
            'dist_rtg_gate': dist_rtg_gate,
            'rtg_energy': rtg_time * 1.0,  # e^rtg_i
            'yard_energy': yard_time * 0.8,  # e^yard_i
            'wait_penalty_weight': random.uniform(0.6, 1.8)  # w_i
        })

    return {
        'tasks': tasks,
        'num_rtg': NUM_RTG,
        'num_yard': NUM_YARD,
        'num_trucks': NUM_TRUCKS,
        'robust_budget': ROBUST_BUDGET
    }


# ====================== 3. 染色体编码与解码（完全对应第三章决策变量） ======================
class Chromosome:
    def __init__(self, rtg_alloc: List[int], yard_truck_alloc: List[int], task_order: List[int]):
        self.rtg_alloc = rtg_alloc  # 对应x_ir：任务i分配给火车正面吊r
        self.yard_truck_alloc = yard_truck_alloc  # 对应u_iy和y_iv：复合编码
        self.task_order = task_order  # 对应z^R_ij, z^Y_ij, z^V_ij：任务优先级
        self.fitness = None  # (f1, f2) = (T_max, 总能耗)
        self.rank = None  # 非支配层级
        self.crowding_distance = None  # 拥挤度距离
        self.domination_count = 0  # 支配计数
        self.dominated_solutions = []  # 被支配的解列表


def decode_chromosome(chromosome: Chromosome, instance: Dict) -> Tuple[Dict, float, float]:
    """
    完全对应第三章3.3节的解码过程，包含确定性基础模型约束与第一层鲁棒性保护
    返回：(调度方案字典, f1总完工时间, E_total基础总能耗)
    """
    tasks = instance['tasks']
    num_rtg = instance['num_rtg']
    num_yard = instance['num_yard']
    num_trucks = instance['num_trucks']
    robust_budget = instance['robust_budget']

    # 1. 解析资源分配（对应第三章3.3.2任务分配约束）
    rtg_map = {}  # task_id -> rtg_id
    yard_map = {}  # task_id -> yard_id (-1表示无)
    truck_map = {}  # task_id -> truck_id
    for i in range(NUM_TASKS):
        task_id = i
        rtg_map[task_id] = chromosome.rtg_alloc[i]
        # 解析复合编码：yard_truck_alloc[i] = truck_id + num_trucks * yard_id
        alloc_val = chromosome.yard_truck_alloc[i]
        truck_id = alloc_val % num_trucks
        yard_id = alloc_val // num_trucks
        truck_map[task_id] = truck_id
        # 校验堆场分配合法性（B/D类无堆场作业）
        if tasks[task_id]['type'] in ['B', 'D']:
            yard_map[task_id] = -1
        else:
            yard_map[task_id] = yard_id % num_yard

    # 2. 构建设备任务序列（按优先级排列，对应顺序变量z）
    rtg_sequences = {r: [] for r in range(num_rtg)}
    yard_sequences = {y: [] for y in range(num_yard)}
    truck_sequences = {v: [] for v in range(num_trucks)}
    for task_id in chromosome.task_order:
        rtg = rtg_map[task_id]
        rtg_sequences[rtg].append(task_id)
        truck = truck_map[task_id]
        truck_sequences[truck].append(task_id)
        yard = yard_map[task_id]
        if yard != -1:
            yard_sequences[yard].append(task_id)

    # 3. 递推计算作业时间（包含第一层鲁棒性时间保护）
    rtg_available = {r: 0.0 for r in range(num_rtg)}
    yard_available = {y: 0.0 for y in range(num_yard)}
    truck_available = {v: 0.0 for v in range(num_trucks)}
    task_times = {}  # task_id -> (s_rtg, c_rtg, s_yard, c_yard, s_truck, c_truck, wait_time)

    # 保存详细调度过程用于文字输出
    rtg_schedule_details = {r: [] for r in range(num_rtg)}
    yard_schedule_details = {y: [] for y in range(num_yard)}
    truck_schedule_details = {v: [] for v in range(num_trucks)}

    total_wait_time = 0.0
    total_energy = 0.0
    # 记录已使用的鲁棒预算（用于第一层作业时间不确定性保护）
    rtg_used_budget = {r: 0 for r in range(num_rtg)}

    for task_id in chromosome.task_order:
        task = tasks[task_id]
        rtg = rtg_map[task_id]
        truck = truck_map[task_id]
        yard = yard_map[task_id]
        wait_time = 0.0

        # 根据第三章3.1.2的五种任务类型分别处理时序约束
        if task['type'] == 'A':  # 进口箱（需堆存）：火车卸->集卡运->堆场卸
            # 第二层：弹性时间窗（仅外部集卡，A类为内部集卡，无严格窗但有鲁棒缓冲）
            s_truck = max(rtg_available[rtg], truck_available[truck])
            # 第一层：作业时间鲁棒保护（若预算未用完，增加时间缓冲）
            rtg_buffer = task['rtg_time_dev'] if rtg_used_budget[rtg] < robust_budget else 0.0
            if rtg_buffer > 0:
                rtg_used_budget[rtg] += 1

            s_rtg = s_truck
            c_rtg = s_rtg + task['rtg_time'] + rtg_buffer
            rtg_available[rtg] = c_rtg

            move_time_truck = task['dist_rtg_yard'] / TRUCK_SPEED
            c_truck = s_truck + move_time_truck
            truck_available[truck] = c_truck

            s_yard = max(c_truck, yard_available[yard])
            c_yard = s_yard + task['yard_time']
            yard_available[yard] = c_yard

            # 累加能耗（对应第三章3.3.1）
            total_energy += task['rtg_energy'] + task['yard_energy'] + move_time_truck * TRUCK_UNIT_DIST_ENERGY

            # 保存调度详情
            rtg_schedule_details[rtg].append((task_id, s_rtg, c_rtg, 'A-卸火车', rtg_buffer))
            truck_schedule_details[truck].append((task_id, s_truck, c_truck, 'A-运箱至堆场', 0.0))
            yard_schedule_details[yard].append((task_id, s_yard, c_yard, 'A-卸至堆场', 0.0))

        elif task['type'] == 'B':  # 进口箱（直接出港）：火车卸->外部集卡运出
            # 第二层：弹性时间窗约束（对应第三章3.4.2）
            earliest_start = max(rtg_available[rtg], truck_available[truck],
                                 task['time_window'][0] - task['delta_time'])
            latest_start = task['time_window'][1] + task['delta_time']
            s_truck = max(earliest_start, 0.0)
            s_truck = min(s_truck, latest_start)

            # 等待时间计算（对应第三章3.4.2）
            wait_time = max(task['time_window'][0] - s_truck, 0.0)
            total_wait_time += wait_time * task['wait_penalty_weight']

            # 第一层：鲁棒时间保护
            rtg_buffer = task['rtg_time_dev'] if rtg_used_budget[rtg] < robust_budget else 0.0
            if rtg_buffer > 0:
                rtg_used_budget[rtg] += 1

            s_rtg = s_truck + wait_time
            c_rtg = s_rtg + task['rtg_time'] + rtg_buffer
            rtg_available[rtg] = c_rtg

            move_time_truck = task['dist_rtg_gate'] / TRUCK_SPEED
            c_truck = s_truck + wait_time + move_time_truck
            truck_available[truck] = c_truck

            total_energy += task['rtg_energy'] + move_time_truck * TRUCK_UNIT_DIST_ENERGY
            s_yard, c_yard = 0.0, 0.0

            rtg_schedule_details[rtg].append((task_id, s_rtg, c_rtg, 'B-卸火车', rtg_buffer))
            truck_schedule_details[truck].append((task_id, s_truck + wait_time, c_truck, 'B-运出港', wait_time))

        elif task['type'] == 'C':  # 出口箱（从堆场取）：堆场装->集卡运->火车装
            s_yard = max(yard_available[yard], truck_available[truck])
            c_yard = s_yard + task['yard_time']
            yard_available[yard] = c_yard

            s_truck = s_yard
            move_time_truck = task['dist_rtg_yard'] / TRUCK_SPEED
            c_truck = s_truck + move_time_truck
            truck_available[truck] = c_truck

            # 第一层：鲁棒时间保护
            rtg_buffer = task['rtg_time_dev'] if rtg_used_budget[rtg] < robust_budget else 0.0
            if rtg_buffer > 0:
                rtg_used_budget[rtg] += 1

            s_rtg = max(c_truck, rtg_available[rtg])
            c_rtg = s_rtg + task['rtg_time'] + rtg_buffer
            rtg_available[rtg] = c_rtg

            total_energy += task['rtg_energy'] + task['yard_energy'] + move_time_truck * TRUCK_UNIT_DIST_ENERGY

            yard_schedule_details[yard].append((task_id, s_yard, c_yard, 'C-从堆场取', 0.0))
            truck_schedule_details[truck].append((task_id, s_truck, c_truck, 'C-运至火车线', 0.0))
            rtg_schedule_details[rtg].append((task_id, s_rtg, c_rtg, 'C-装火车', rtg_buffer))

        elif task['type'] == 'D':  # 出口箱（直接到港）：外部集卡到->火车装
            # 第二层：弹性时间窗
            earliest_start = max(truck_available[truck], task['time_window'][0] - task['delta_time'])
            latest_start = task['time_window'][1] + task['delta_time']
            s_truck = max(earliest_start, 0.0)
            s_truck = min(s_truck, latest_start)

            wait_time = max(task['time_window'][0] - s_truck, 0.0)
            total_wait_time += wait_time * task['wait_penalty_weight']

            # 第一层：鲁棒时间保护
            rtg_buffer = task['rtg_time_dev'] if rtg_used_budget[rtg] < robust_budget else 0.0
            if rtg_buffer > 0:
                rtg_used_budget[rtg] += 1

            s_rtg = max(s_truck + wait_time, rtg_available[rtg])
            c_rtg = s_rtg + task['rtg_time'] + rtg_buffer
            rtg_available[rtg] = c_rtg

            move_time_truck = task['dist_rtg_gate'] / TRUCK_SPEED
            c_truck = s_truck + wait_time + move_time_truck
            truck_available[truck] = c_truck

            total_energy += task['rtg_energy'] + move_time_truck * TRUCK_UNIT_DIST_ENERGY
            s_yard, c_yard = 0.0, 0.0

            truck_schedule_details[truck].append((task_id, s_truck + wait_time, c_truck, 'D-运进港', wait_time))
            rtg_schedule_details[rtg].append((task_id, s_rtg, c_rtg, 'D-装火车', rtg_buffer))

        elif task['type'] == 'E':  # 空箱：类似A类，存空箱区
            s_truck = max(rtg_available[rtg], truck_available[truck])

            rtg_buffer = task['rtg_time_dev'] if rtg_used_budget[rtg] < robust_budget else 0.0
            if rtg_buffer > 0:
                rtg_used_budget[rtg] += 1

            s_rtg = s_truck
            c_rtg = s_rtg + task['rtg_time'] + rtg_buffer
            rtg_available[rtg] = c_rtg

            move_time_truck = task['dist_rtg_yard'] / TRUCK_SPEED
            c_truck = s_truck + move_time_truck
            truck_available[truck] = c_truck

            s_yard = max(c_truck, yard_available[yard])
            c_yard = s_yard + task['yard_time']
            yard_available[yard] = c_yard

            total_energy += task['rtg_energy'] + task['yard_energy'] + move_time_truck * TRUCK_UNIT_DIST_ENERGY

            rtg_schedule_details[rtg].append((task_id, s_rtg, c_rtg, 'E-卸火车(空)', rtg_buffer))
            truck_schedule_details[truck].append((task_id, s_truck, c_truck, 'E-运空箱', 0.0))
            yard_schedule_details[yard].append((task_id, s_yard, c_yard, 'E-存空箱区', 0.0))

        task_times[task_id] = (s_rtg, c_rtg, s_yard, c_yard, s_truck, c_truck, wait_time)

    # 计算总完工时间f1（对应第三章3.3.1式3-4）
    f1 = max(rtg_available.values())
    # 计算基础总能耗E_total（对应第三章3.3.1式3-5至3-8，简化空载能耗为固定比例）
    E_total = total_energy + total_wait_time + f1 * (
                RTG_IDLE_ENERGY * num_rtg + YARD_IDLE_ENERGY * num_yard + TRUCK_IDLE_ENERGY * num_trucks) * 0.15

    schedule = {
        'rtg_map': rtg_map,
        'yard_map': yard_map,
        'truck_map': truck_map,
        'task_times': task_times,
        'rtg_schedule': rtg_schedule_details,
        'yard_schedule': yard_schedule_details,
        'truck_schedule': truck_schedule_details,
        'rtg_available': rtg_available,
        'yard_available': yard_available,
        'truck_available': truck_available
    }
    return schedule, f1, E_total


# ====================== 4. 文字版详细调度过程打印 ======================
def print_schedule_details(schedule: Dict, instance: Dict, f1: float, f2: float):
    """完全对应第三章模型的文字版调度详情输出"""
    tasks = instance['tasks']
    print("\n" + "=" * 100)
    print("陆港集装箱装卸资源协同调度详细方案")
    print("=" * 100)
    print(f"优化目标值：总完工时间 f1 = {f1:.2f} | 总能耗 f2 = {f2:.2f}")
    print("=" * 100)

    # 1. 火车线正面吊调度（核心设备）
    print("\n【1. 火车线正面吊调度详情（含第一层鲁棒时间缓冲）】")
    print("-" * 100)
    for rtg_id in sorted(schedule['rtg_schedule'].keys()):
        print(f"\n  ▶ 火车线正面吊 ID: {rtg_id}")
        print(
            f"    {'任务ID':<8} {'类型':<6} {'操作':<15} {'开始时间':<10} {'完成时间':<10} {'持续时间':<10} {'鲁棒缓冲':<10}")
        print("    " + "-" * 85)
        for task in schedule['rtg_schedule'][rtg_id]:
            task_id, s, c, op, buffer = task
            task_type = tasks[task_id]['type']
            print(f"    {task_id:<8} {task_type:<6} {op:<15} {s:<10.2f} {c:<10.2f} {c - s:<10.2f} {buffer:<10.2f}")

    # 2. 堆场正面吊调度
    print("\n【2. 堆场正面吊调度详情】")
    print("-" * 100)
    for yard_id in sorted(schedule['yard_schedule'].keys()):
        print(f"\n  ▶ 堆场正面吊 ID: {yard_id}")
        print(f"    {'任务ID':<8} {'类型':<6} {'操作':<15} {'开始时间':<10} {'完成时间':<10} {'持续时间':<10}")
        print("    " + "-" * 70)
        for task in schedule['yard_schedule'][yard_id]:
            task_id, s, c, op, _ = task
            task_type = tasks[task_id]['type']
            print(f"    {task_id:<8} {task_type:<6} {op:<15} {s:<10.2f} {c:<10.2f} {c - s:<10.2f}")

    # 3. 集卡调度（含内外部集卡区分）
    print("\n【3. 集卡调度详情（含第二层弹性时间窗等待时间）】")
    print("-" * 100)
    for truck_id in sorted(schedule['truck_schedule'].keys()):
        print(f"\n  ▶ 集卡 ID: {truck_id}")
        print(
            f"    {'任务ID':<8} {'类型':<6} {'操作':<15} {'开始时间':<10} {'完成时间':<10} {'持续时间':<10} {'等待时间':<10}")
        print("    " + "-" * 80)
        for task in schedule['truck_schedule'][truck_id]:
            task_id, s, c, op, wait = task
            task_type = tasks[task_id]['type']
            print(f"    {task_id:<8} {task_type:<6} {op:<15} {s:<10.2f} {c:<10.2f} {c - s:<10.2f} {wait:<10.2f}")

    print("\n" + "=" * 100)
    print("调度方案说明：")
    print("  - 鲁棒缓冲：针对作业时间不确定性的第一层保护时间（第三章3.4.1）")
    print("  - 等待时间：针对外部集卡到达扰动的第二层弹性时间窗等待（第三章3.4.2）")
    print("  - 总能耗已包含第三层设备故障期望损失的惩罚项（第三章3.4.3）")
    print("=" * 100)


# ====================== 5. 初始种群生成（混合启发式，对应第四章4.3） ======================
def generate_initial_chromosome(instance: Dict, method: str = 'random') -> Chromosome:
    """生成单个初始染色体，支持多种启发式规则"""
    tasks = instance['tasks']
    num_rtg = instance['num_rtg']
    num_yard = instance['num_yard']
    num_trucks = instance['num_trucks']

    if method == 'random':
        # 随机生成
        rtg_alloc = [random.randint(0, num_rtg - 1) for _ in range(NUM_TASKS)]
        yard_truck_alloc = []
        for i in range(NUM_TASKS):
            truck_id = random.randint(0, num_trucks - 1)
            yard_id = random.randint(0, num_yard - 1) if tasks[i]['type'] not in ['B', 'D'] else 0
            yard_truck_alloc.append(truck_id + num_trucks * yard_id)
        task_order = list(range(NUM_TASKS))
        random.shuffle(task_order)

    elif method == 'EDD':
        # 最早截止时间优先（针对外部集卡任务）
        task_order = sorted(range(NUM_TASKS),
                            key=lambda x: tasks[x]['time_window'][1] if tasks[x]['type'] in ['B', 'D'] else tasks[x][
                                'rtg_time'])
        rtg_alloc = [i % num_rtg for i in range(NUM_TASKS)]
        yard_truck_alloc = []
        for i in range(NUM_TASKS):
            truck_id = i % num_trucks
            yard_id = i % num_yard if tasks[i]['type'] not in ['B', 'D'] else 0
            yard_truck_alloc.append(truck_id + num_trucks * yard_id)

    elif method == 'SPT':
        # 最短作业时间优先
        task_order = sorted(range(NUM_TASKS), key=lambda x: tasks[x]['rtg_time'])
        rtg_alloc = [i % num_rtg for i in range(NUM_TASKS)]
        yard_truck_alloc = []
        for i in range(NUM_TASKS):
            truck_id = i % num_trucks
            yard_id = i % num_yard if tasks[i]['type'] not in ['B', 'D'] else 0
            yard_truck_alloc.append(truck_id + num_trucks * yard_id)

    elif method == 'energy_greedy':
        # 能耗贪婪优先
        task_order = sorted(range(NUM_TASKS), key=lambda x: (tasks[x]['rtg_energy'] + tasks[x]['yard_energy']))
        rtg_alloc = [i % num_rtg for i in range(NUM_TASKS)]
        yard_truck_alloc = []
        for i in range(NUM_TASKS):
            truck_id = i % num_trucks
            yard_id = i % num_yard if tasks[i]['type'] not in ['B', 'D'] else 0
            yard_truck_alloc.append(truck_id + num_trucks * yard_id)

    return Chromosome(rtg_alloc, yard_truck_alloc, task_order)


def generate_initial_population(instance: Dict) -> List[Chromosome]:
    """生成初始种群：30%启发式，70%随机（对应第四章4.3）"""
    population = []
    num_heuristic = int(POPULATION_SIZE * 0.3)
    # 四种启发式规则各生成等量个体
    for _ in range(num_heuristic // 4):
        population.append(generate_initial_chromosome(instance, 'EDD'))
        population.append(generate_initial_chromosome(instance, 'SPT'))
        population.append(generate_initial_chromosome(instance, 'energy_greedy'))
        population.append(generate_initial_chromosome(instance, 'random'))
    # 补充随机个体至种群规模
    while len(population) < POPULATION_SIZE:
        population.append(generate_initial_chromosome(instance, 'random'))
    return population


# ====================== 6. 遗传算子设计（对应第四章4.4） ======================
def selection(population: List[Chromosome]) -> Chromosome:
    """二元锦标赛选择（对应第四章4.4.1）"""
    ind1 = random.choice(population)
    ind2 = random.choice(population)
    if ind1.rank < ind2.rank:
        return ind1
    elif ind1.rank > ind2.rank:
        return ind2
    else:
        return ind1 if ind1.crowding_distance > ind2.crowding_distance else ind2


def crossover(parent1: Chromosome, parent2: Chromosome, instance: Dict) -> Tuple[Chromosome, Chromosome]:
    """
    交叉算子：分配段均匀交叉，排序段OX顺序交叉（对应第四章4.4.2）
    """
    num_trucks = instance['num_trucks']
    num_yard = instance['num_yard']

    # 1. 分配段均匀交叉
    mask = [random.random() < 0.5 for _ in range(NUM_TASKS)]
    child1_rtg = [parent1.rtg_alloc[i] if mask[i] else parent2.rtg_alloc[i] for i in range(NUM_TASKS)]
    child2_rtg = [parent2.rtg_alloc[i] if mask[i] else parent1.rtg_alloc[i] for i in range(NUM_TASKS)]

    child1_yard_truck = [parent1.yard_truck_alloc[i] if mask[i] else parent2.yard_truck_alloc[i] for i in
                         range(NUM_TASKS)]
    child2_yard_truck = [parent2.yard_truck_alloc[i] if mask[i] else parent1.yard_truck_alloc[i] for i in
                         range(NUM_TASKS)]

    # 2. 排序段OX顺序交叉（保证全排列合法性）
    def ox_crossover(order1, order2):
        size = len(order1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = order1[start:end]
        ptr = end
        for task in order2:
            if task not in child:
                if ptr >= size:
                    ptr = 0
                child[ptr] = task
                ptr += 1
        return child

    child1_order = ox_crossover(parent1.task_order, parent2.task_order)
    child2_order = ox_crossover(parent2.task_order, parent1.task_order)

    return Chromosome(child1_rtg, child1_yard_truck, child1_order), Chromosome(child2_rtg, child2_yard_truck,
                                                                               child2_order)


def mutation(chromosome: Chromosome, instance: Dict) -> Chromosome:
    """
    变异算子：分配段随机重分配，排序段双交换/逆序（对应第四章4.4.3）
    """
    tasks = instance['tasks']
    num_rtg = instance['num_rtg']
    num_yard = instance['num_yard']
    num_trucks = instance['num_trucks']

    mutated = copy.deepcopy(chromosome)

    # 1. 分配段变异
    for i in range(NUM_TASKS):
        if random.random() < MUTATION_PROB:
            # 火车正面吊重分配
            mutated.rtg_alloc[i] = random.randint(0, num_rtg - 1)
            # 堆场正面吊+集卡重分配
            new_truck = random.randint(0, num_trucks - 1)
            new_yard = random.randint(0, num_yard - 1) if tasks[i]['type'] not in ['B', 'D'] else 0
            mutated.yard_truck_alloc[i] = new_truck + num_trucks * new_yard

    # 2. 排序段变异（1:1概率选择双交换或逆序）
    if random.random() < MUTATION_PROB:
        if random.random() < 0.5:
            # 双交换变异
            i, j = random.sample(range(NUM_TASKS), 2)
            mutated.task_order[i], mutated.task_order[j] = mutated.task_order[j], mutated.task_order[i]
        else:
            # 逆序变异
            start, end = sorted(random.sample(range(NUM_TASKS), 2))
            mutated.task_order[start:end] = reversed(mutated.task_order[start:end])

    return mutated


# ====================== 7. 内层蒙特卡洛仿真评估（对应第三章3.4.3与第四章4.5） ======================
def monte_carlo_simulation(schedule: Dict, E_total: float, instance: Dict) -> float:
    """
    完全对应第三章3.4.3的设备故障期望能耗损失评估
    返回：期望故障能耗损失E[L(π)]
    """
    total_loss = 0.0
    tasks = instance['tasks']

    for _ in range(SIMULATION_SCENARIOS):
        # 1. 复制原调度方案
        sim_rtg_available = schedule['rtg_available'].copy()
        sim_yard_available = schedule['yard_available'].copy()
        sim_truck_available = schedule['truck_available'].copy()
        T_max = max(sim_rtg_available.values())

        # 2. 生成故障情景（对应第三章3.4.3故障情景生成器）
        failure_events = []

        # 火车正面吊故障
        for r in range(NUM_RTG):
            t = 0.0
            while t < T_max:
                delta_t = random.expovariate(RTG_FAILURE_RATE)
                t += delta_t
                if t < T_max:
                    repair_time = max(0.1, random.normalvariate(REPAIR_TIME_MEAN, REPAIR_TIME_STD))
                    failure_events.append(('rtg', r, t, repair_time))

        # 堆场正面吊故障
        for y in range(NUM_YARD):
            t = 0.0
            while t < T_max:
                delta_t = random.expovariate(YARD_FAILURE_RATE)
                t += delta_t
                if t < T_max:
                    repair_time = max(0.1, random.normalvariate(REPAIR_TIME_MEAN * 0.9, REPAIR_TIME_STD))
                    failure_events.append(('yard', y, t, repair_time))

        # 集卡故障
        for v in range(NUM_TRUCKS):
            t = 0.0
            while t < T_max:
                delta_t = random.expovariate(TRUCK_FAILURE_RATE)
                t += delta_t
                if t < T_max:
                    repair_time = max(0.1, random.normalvariate(REPAIR_TIME_MEAN * 0.8, REPAIR_TIME_STD))
                    failure_events.append(('truck', v, t, repair_time))

        # 按时间排序故障事件
        failure_events.sort(key=lambda x: x[2])

        # 3. 快速重调度（对应第三章3.4.3“最早可用-最低增量能耗”规则）
        extra_energy = 0.0
        for event in failure_events:
            eq_type, eq_id, t_f, d_f = event

            # 设备故障期间闲置能耗
            if eq_type == 'rtg':
                extra_energy += d_f * RTG_IDLE_ENERGY
                sim_rtg_available[eq_id] = max(sim_rtg_available[eq_id], t_f + d_f)
            elif eq_type == 'yard':
                extra_energy += d_f * YARD_IDLE_ENERGY
                sim_yard_available[eq_id] = max(sim_yard_available[eq_id], t_f + d_f)
            else:
                extra_energy += d_f * TRUCK_IDLE_ENERGY
                sim_truck_available[eq_id] = max(sim_truck_available[eq_id], t_f + d_f)

        # 4. 计算故障情景下的总能耗增量（简化为重调度后延迟带来的额外闲置能耗）
        new_T_max = max(max(sim_rtg_available.values()), max(sim_yard_available.values()),
                        max(sim_truck_available.values()))
        extra_energy += (new_T_max - T_max) * (
                    RTG_IDLE_ENERGY * NUM_RTG + YARD_IDLE_ENERGY * NUM_YARD + TRUCK_IDLE_ENERGY * NUM_TRUCKS) * 0.6

        # 累加该情景的损失
        loss = extra_energy
        total_loss += loss

    # 计算期望损失（对应第三章3.4.3式3-55）
    expected_loss = total_loss / SIMULATION_SCENARIOS
    return expected_loss


# ====================== 8. NSGA-II核心：非支配排序与拥挤度（对应第四章4.6） ======================
def fast_non_dominated_sort(population: List[Chromosome]) -> List[List[Chromosome]]:
    """快速非支配排序（对应第四章4.6.1）"""
    fronts = [[]]
    # 初始化每个个体的支配计数和被支配列表
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in population:
            if p == q:
                continue
            # 判断支配关系：p支配q当且仅当p的所有目标都不劣于q，且至少一个更优
            p_dominates = (p.fitness[0] <= q.fitness[0] and p.fitness[1] <= q.fitness[1]) and \
                          (p.fitness[0] < q.fitness[0] or p.fitness[1] < q.fitness[1])
            q_dominates = (q.fitness[0] <= p.fitness[0] and q.fitness[1] <= p.fitness[1]) and \
                          (q.fitness[0] < p.fitness[0] or q.fitness[1] < p.fitness[1])
            if p_dominates:
                p.dominated_solutions.append(q)
            elif q_dominates:
                p.domination_count += 1
        # 第一非支配层级
        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)

    # 构建后续层级
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]  # 移除空的最后一层


def crowding_distance_assignment(front: List[Chromosome]):
    """拥挤度距离计算（对应第四章4.6.2）"""
    if len(front) == 0:
        return
    # 初始化所有个体的拥挤度距离为0
    for p in front:
        p.crowding_distance = 0.0
    # 对每个目标函数分别计算
    for obj_idx in [0, 1]:
        # 按当前目标函数值升序排序
        front.sort(key=lambda x: x.fitness[obj_idx])
        # 边界个体的拥挤度距离设为无穷大
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        if len(front) <= 2:
            continue
        # 计算目标函数的取值范围
        f_min = front[0].fitness[obj_idx]
        f_max = front[-1].fitness[obj_idx]
        if f_max == f_min:
            continue  # 避免除以0
        # 计算中间个体的拥挤度分量
        for i in range(1, len(front) - 1):
            front[i].crowding_distance += (front[i + 1].fitness[obj_idx] - front[i - 1].fitness[obj_idx]) / (
                        f_max - f_min)


# ====================== 9. 主算法流程（完全对应第四章4.7） ======================
def main():
    # 1. 生成问题实例
    print("正在生成陆港问题实例...")
    instance = generate_problem_instance()
    print(f"实例生成完成：任务数 {NUM_TASKS}，火车正面吊 {NUM_RTG}，堆场正面吊 {NUM_YARD}，集卡 {NUM_TRUCKS}")
    print(f"鲁棒预算 Γ1 = {ROBUST_BUDGET}，风险厌恶系数 β = {RISK_AVERSION_COEF}")

    # 2. 初始化种群
    print("\n正在生成初始种群...")
    population = generate_initial_population(instance)
    print(f"初始种群规模：{len(population)}")

    # 3. 评估初始种群
    print("\n正在评估初始种群...")
    for ind_idx, ind in enumerate(population):
        if (ind_idx + 1) % 20 == 0:
            print(f"  已评估 {ind_idx + 1}/{len(population)} 个个体")
        schedule, f1, E_total = decode_chromosome(ind, instance)
        expected_loss = monte_carlo_simulation(schedule, E_total, instance)
        # 计算总能耗目标f2（对应第三章3.5.1式3-59）
        f2 = E_total + RISK_AVERSION_COEF * expected_loss
        ind.fitness = (f1, f2)

    # 4. 迭代进化（对应第四章4.7算法流程）
    print("\n开始迭代进化...")
    for iter in range(MAX_ITERATIONS):
        print(f"\n=== 迭代 {iter + 1}/{MAX_ITERATIONS} ===")

        # 4.1 非支配排序与拥挤度计算
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance_assignment(front)

        # 4.2 生成子代种群
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)
            if random.random() < CROSSOVER_PROB:
                child1, child2 = crossover(parent1, parent2, instance)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            child1 = mutation(child1, instance)
            child2 = mutation(child2, instance)
            offspring.append(child1)
            offspring.append(child2)
        offspring = offspring[:POPULATION_SIZE]

        # 4.3 评估子代种群
        for ind in offspring:
            schedule, f1, E_total = decode_chromosome(ind, instance)
            expected_loss = monte_carlo_simulation(schedule, E_total, instance)
            f2 = E_total + RISK_AVERSION_COEF * expected_loss
            ind.fitness = (f1, f2)

        # 4.4 精英保留种群更新（对应第四章4.6.3）
        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)
        new_population = []
        i = 0
        # 按层级依次加入，直到种群规模接近上限
        while len(new_population) + len(fronts[i]) <= POPULATION_SIZE:
            crowding_distance_assignment(fronts[i])
            new_population.extend(fronts[i])
            i += 1
        # 若仍有空间，从当前层级选择拥挤度最大的个体填满
        if len(new_population) < POPULATION_SIZE:
            crowding_distance_assignment(fronts[i])
            fronts[i].sort(key=lambda x: (-x.rank, -x.crowding_distance))
            new_population.extend(fronts[i][:POPULATION_SIZE - len(new_population)])
        population = new_population

        # 输出当前迭代的统计信息
        first_front = [ind for ind in population if ind.rank == 0]
        print(f"当前帕累托前沿解数量：{len(first_front)}")
        if first_front:
            best_f1 = min(ind.fitness[0] for ind in first_front)
            best_f2 = min(ind.fitness[1] for ind in first_front)
            avg_f1 = sum(ind.fitness[0] for ind in first_front) / len(first_front)
            avg_f2 = sum(ind.fitness[1] for ind in first_front) / len(first_front)
            print(f"前沿最优 f1(总完工时间): {best_f1:.2f}, 最优 f2(总能耗): {best_f2:.2f}")
            print(f"前沿平均 f1: {avg_f1:.2f}, 平均 f2: {avg_f2:.2f}")

    # 5. 输出最终结果
    print("\n" + "=" * 100)
    print("算法运行结束！")
    print("=" * 100)

    # 获取最终帕累托前沿
    final_fronts = fast_non_dominated_sort(population)
    final_pareto = final_fronts[0]
    print(f"\n最终帕累托最优解数量：{len(final_pareto)}")

    # 输出所有帕累托解的目标值
    print("\n【帕累托前沿解目标值列表】")
    print("-" * 50)
    print(f"{'解编号':<8} {'总完工时间 f1':<20} {'总能耗 f2':<20}")
    print("-" * 50)
    for idx, ind in enumerate(final_pareto):
        print(f"{idx + 1:<8} {ind.fitness[0]:<20.2f} {ind.fitness[1]:<20.2f}")

    # 选择帕累托前沿中f1最小的解展示详细调度过程
    print("\n\n正在展示【总完工时间最短】的调度方案详细过程...")
    best_ind_f1 = min(final_pareto, key=lambda x: x.fitness[0])
    best_schedule, best_f1, best_E_total = decode_chromosome(best_ind_f1, instance)
    best_expected_loss = monte_carlo_simulation(best_schedule, best_E_total, instance)
    best_f2 = best_E_total + RISK_AVERSION_COEF * best_expected_loss

    # 打印详细调度方案
    print_schedule_details(best_schedule, instance, best_f1, best_f2)


if __name__ == "__main__":
    main()