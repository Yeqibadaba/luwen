import numpy as np
import random
from typing import List, Tuple, Dict
import copy
import time  # 新增：用于计时

# ==========================================
# 1. 全局可配置参数（在此处直接修改数据）
# ==========================================
# --------------------------
# 算法控制参数
# --------------------------
POP_SIZE = 200          # 种群规模 (小规模50, 中规模100, 大规模200)
MAX_ITER = 500             # 最大迭代次数
PC = 0.8                  # 交叉概率
PM = 0.1                  # 变异概率
TOURNAMENT_SIZE = 3       # 锦标赛选择规模
ELITE_ARCHIVE_SIZE = 100  # 外部精英档案最大规模

# --------------------------
# 模型鲁棒参数
# --------------------------
GAMMA_1 = 3               # 第一层鲁棒预算 (作业时间波动保护)
BETA = 1.5                # 风险厌恶系数 (β≥1)
ALPHA = 0.95              # CVaR置信水平 (通常0.95)
DELTA_EARLY = 15          # 外部集卡最大允许早到偏差 (分钟)
DELTA_LATE = 30           # 外部集卡最大允许晚到偏差 (分钟)

# --------------------------
# 设备参数 (可根据实际陆港数据修改)
# --------------------------
# 铁路正面吊 (R)
NUM_R = 2                 # 铁路正面吊数量
V_R = 0.5                 # 铁路正面吊移动速度 (单位/分钟)
E_RTG = 15.0              # 铁路正面吊作业单位功率 (kW)
E_R_MOVE = 8.0            # 铁路正面吊空载移动功率 (kW)
E_R_STANDBY = 2.0         # 铁路正面吊待机功率 (kW)
E_R_IDLE = 2.5            # 铁路正面吊故障闲置功率 (kW)
LAMBDA_R = 0.002          # 铁路正面吊故障率 (次/分钟)

# 堆场正面吊 (Y)
NUM_Y = 2                 # 堆场正面吊数量
V_Y = 0.4                 # 堆场正面吊移动速度 (单位/分钟)
E_YARD = 12.0             # 堆场正面吊作业单位功率 (kW)
E_Y_MOVE = 6.0            # 堆场正面吊空载移动功率 (kW)
E_Y_STANDBY = 1.5         # 堆场正面吊待机功率 (kW)
E_Y_IDLE = 2.0            # 堆场正面吊故障闲置功率 (kW)
LAMBDA_Y = 0.0015         # 堆场正面吊故障率 (次/分钟)

# 集卡 (V)
NUM_V_IN = 3              # 内部集卡数量
NUM_V_OUT = 2             # 外部集卡数量
V_T_LOAD = 0.6            # 集卡带箱移动速度 (单位/分钟)
V_T_EMPTY = 0.8           # 集卡空驶移动速度 (单位/分钟)
E_T_LOAD = 0.15           # 集卡带箱单位距离油耗 (L/单位)
E_T_EMPTY = 0.08          # 集卡空驶单位距离油耗 (L/单位)
E_T_STANDBY = 0.05        # 集卡怠速油耗 (L/分钟)
E_T_IDLE = 0.06           # 集卡故障闲置油耗 (L/分钟)
LAMBDA_V = 0.001          # 集卡故障率 (次/分钟)

# 成本参数
C_ELEC = 0.8               # 电价 (元/kWh)
C_DIESEL = 7.5             # 柴油价 (元/L)
W_EARLY = 5.0              # 外部集卡早到单位时间惩罚 (元/分钟)
W_LATE = 15.0              # 外部集卡晚到单位时间惩罚 (元/分钟)

# 设备启停约束
T1 = 10                    # 正面吊待机/停机切换阈值 (分钟)
T2 = 8                     # 集卡怠速/熄火切换阈值 (分钟)
N_MAX = 5                  # 单台设备最大启停次数

# --------------------------
# 任务参数 (可在此处修改任务数据)
# --------------------------
NUM_TASKS = 200             # 总任务数
# 任务类型比例: A(进口需堆存), B(进口直出), C(出口堆场取), D(出口直进), E(空箱)
TASK_TYPE_RATIO = [0.25, 0.2, 0.25, 0.2, 0.1]
# 堆场分区 (对应论文3.1.2节)
ZONES = {1: "暂存区", 2: "堆存区", 3: "出口箱区", 4: "空箱区"}
ZONE_COORDS = {            # 堆场分区中心点坐标 (x, y)
    1: (5, 10),
    2: (15, 20),
    3: (10, 30),
    4: (20, 5)
}
RAIL_LINE_COORDS = {i: (0, i*2) for i in range(1, 6)}  # 铁路装卸线作业点坐标
GATE_COORD = (25, 15)      # 港外出入口坐标

# 作业时间波动参数
P_RTG_MAX_DEV = 0.2        # 铁路正面吊作业时间最大波动比例 (±20%)
P_YARD_MAX_DEV = 0.15      # 堆场正面吊作业时间最大波动比例 (±15%)

# --------------------------
# 仿真参数
# --------------------------
Q = 100                     # 蒙特卡洛仿真情景数 (迭代初期可用50, 后期200)
SEED = 42                   # 随机种子 (保证可复现)

# ==========================================
# 2. 基础数据结构与辅助函数
# ==========================================
random.seed(SEED)
np.random.seed(SEED)

class Task:
    def __init__(self, task_id: int):
        self.id = task_id
        self.type = random.choices(['A', 'B', 'C', 'D', 'E'], weights=TASK_TYPE_RATIO)[0]
        self.pos = random.choice(list(RAIL_LINE_COORDS.keys()))  # 铁路装卸线作业点
        self.zone = random.choice([1,2,3]) if self.type in ['A','C'] else 4 if self.type == 'E' else None

        # 标准作业时间 (分钟)
        self.p_rtg = np.random.uniform(8, 15)
        self.p_yard = np.random.uniform(7, 12) if self.type in ['A','C','E'] else 0

        # 最大波动值
        self.p_rtg_hat = self.p_rtg * P_RTG_MAX_DEV
        self.p_yard_hat = self.p_yard * P_YARD_MAX_DEV

        # 外部集卡任务时间窗
        if self.type in ['B', 'D']:
            base_time = self.id * 10 + np.random.uniform(0, 30)
            self.a = base_time
            self.b = base_time + 45
        else:
            self.a = None
            self.b = None

class Equipment:
    def __init__(self, eq_id: int, eq_type: str):
        self.id = eq_id
        self.type = eq_type  # 'R', 'Y', 'V_in', 'V_out'
        self.available_time = 0.0
        self.task_sequence = []

class Individual:
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # 三段式编码
        self.r_alloc = [0] * num_tasks          # 段1: 铁路正面吊分配
        self.vy_alloc = [0] * num_tasks         # 段2: 集卡-堆场复合分配
        self.task_order = list(range(num_tasks)) # 段3: 任务优先级排序
        random.shuffle(self.task_order)

        # 解码后的数据
        self.f1 = np.inf  # 目标1: 总完工时间
        self.f2 = np.inf  # 目标2: 综合成本
        self.feasible = True
        self.constraint_violation = 0.0
        self.rank = 0
        self.crowding_distance = 0.0

        # 仿真中间数据
        self.base_schedule = None

    def __eq__(self, other):
        """用于去重：判断两个体是否相同"""
        if not isinstance(other, Individual):
            return False
        return (self.r_alloc == other.r_alloc and
                self.vy_alloc == other.vy_alloc and
                self.task_order == other.task_order)

    def __hash__(self):
        """用于去重：生成个体哈希值"""
        return hash((tuple(self.r_alloc), tuple(self.vy_alloc), tuple(self.task_order)))

def manhattan_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算曼哈顿距离 (对应论文式3-2)"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_task_coords(task: Task) -> Dict[str, Tuple[float, float]]:
    """获取任务相关的坐标点"""
    coords = {}
    coords['rail'] = RAIL_LINE_COORDS[task.pos]
    if task.type in ['A', 'C', 'E']:
        coords['yard'] = ZONE_COORDS[task.zone]
    if task.type in ['B', 'D']:
        coords['gate'] = GATE_COORD
    return coords

# ==========================================
# 3. 编码与解码模块
# ==========================================
def initialize_individual(tasks: List[Task]) -> Individual:
    """随机初始化一个合法个体"""
    ind = Individual(len(tasks))
    num_v = NUM_V_IN + NUM_V_OUT

    # 段1: 铁路正面吊分配 (随机)
    for i in range(ind.num_tasks):
        ind.r_alloc[i] = random.randint(0, NUM_R - 1)

    # 段2: 集卡-堆场复合分配 (遵循类型匹配)
    for i in range(ind.num_tasks):
        task = tasks[i]
        # 集卡分配
        if task.type in ['A', 'C', 'E']:
            v = random.randint(0, NUM_V_IN - 1)  # 内部集卡
        else:
            v = random.randint(NUM_V_IN, num_v - 1)  # 外部集卡

        # 堆场正面吊分配
        if task.type in ['A', 'C', 'E']:
            y = random.randint(0, NUM_Y - 1)
        else:
            y = 0  # 无堆场作业

        ind.vy_alloc[i] = v + num_v * y

    # 段3: 任务优先级已在构造函数中随机打乱
    return ind

def decode(ind: Individual, tasks: List[Task]) -> Tuple[float, float, bool, float, dict]:
    """
    完整解码流程 (对应论文4.2.2节)
    返回: (f1, f2, feasible, constraint_violation, base_schedule)
    """
    num_v = NUM_V_IN + NUM_V_OUT
    schedule = {
        'r': [Equipment(i, 'R') for i in range(NUM_R)],
        'y': [Equipment(i, 'Y') for i in range(NUM_Y)],
        'v': [Equipment(i, 'V_in' if i < NUM_V_IN else 'V_out') for i in range(num_v)]
    }
    task_times = {}  # 存储每个任务的时间节点

    # 步骤1: 资源分配解析
    r_tasks = [[] for _ in range(NUM_R)]
    y_tasks = [[] for _ in range(NUM_Y)]
    v_tasks = [[] for _ in range(num_v)]

    for i in range(ind.num_tasks):
        task = tasks[i]
        r_idx = ind.r_alloc[i]
        r_tasks[r_idx].append(i)

        # 解析复合分配
        vy_code = ind.vy_alloc[i]
        v_idx = vy_code % num_v
        y_idx = vy_code // num_v
        v_tasks[v_idx].append(i)
        if task.type in ['A', 'C', 'E']:
            y_tasks[y_idx].append(i)

    # 步骤2: 按优先级排序构建设备任务序列
    priority_order = ind.task_order
    for eq_list in [r_tasks, y_tasks, v_tasks]:
        for seq in eq_list:
            seq.sort(key=lambda x: priority_order.index(x))

    # 步骤3: 作业时间递推与鲁棒性嵌入
    C_energy = 0.0
    C_penalty = 0.0
    T_max = 0.0
    feasible = True
    violation = 0.0

    # 按任务优先级顺序处理
    for task_id in priority_order:
        task = tasks[task_id]
        coords = get_task_coords(task)

        # 获取分配的设备
        r_idx = ind.r_alloc[task_id]
        vy_code = ind.vy_alloc[task_id]
        v_idx = vy_code % num_v
        y_idx = vy_code // num_v

        r_eq = schedule['r'][r_idx]
        v_eq = schedule['v'][v_idx]
        y_eq = schedule['y'][y_idx] if task.type in ['A','C','E'] else None

        # 初始化时间节点
        s_C = 0.0; c_C = 0.0
        s_Y = 0.0; c_Y = 0.0
        s_T_rail = 0.0; c_T_rail = 0.0
        s_T_yard = 0.0; c_T_yard = 0.0
        W_early = 0.0; W_late = 0.0

        # --- 分任务类型处理时序 (对应论文3.3.2节) ---
        if task.type == 'A':
            # 类型A: 铁路卸->内部集卡->堆场卸
            # 1. 铁路正面吊与集卡就位时间
            rail_move_time = 0.0
            if r_eq.task_sequence:
                last_task = tasks[r_eq.task_sequence[-1]]
                last_coords = get_task_coords(last_task)
                dist = manhattan_distance(last_coords['rail'], coords['rail'])
                rail_move_time = dist / V_R

            s_C = max(r_eq.available_time + rail_move_time, v_eq.available_time)
            # 第一层鲁棒保护: 前Gamma_1个任务加缓冲
            if len(r_eq.task_sequence) < GAMMA_1:
                s_C += task.p_rtg_hat

            c_C = s_C + task.p_rtg
            s_T_rail = s_C
            c_T_rail = c_C

            # 2. 集卡行驶到堆场
            dist_rail_yard = manhattan_distance(coords['rail'], coords['yard'])
            move_time = dist_rail_yard / V_T_LOAD
            s_T_yard = c_T_rail + move_time
            c_T_yard = s_T_yard  # 堆场作业完成时间不早于集卡到达

            # 3. 堆场正面吊作业
            yard_move_time = 0.0
            if y_eq.task_sequence:
                last_task = tasks[y_eq.task_sequence[-1]]
                last_coords = get_task_coords(last_task)
                dist = manhattan_distance(last_coords['yard'], coords['yard'])
                yard_move_time = dist / V_Y

            s_Y = max(y_eq.available_time + yard_move_time, s_T_yard)
            if len(y_eq.task_sequence) < GAMMA_1:
                s_Y += task.p_yard_hat
            c_Y = s_Y + task.p_yard

            # 更新设备可用时间
            r_eq.available_time = c_C
            r_eq.task_sequence.append(task_id)
            v_eq.available_time = c_T_yard
            v_eq.task_sequence.append(task_id)
            y_eq.available_time = c_Y
            y_eq.task_sequence.append(task_id)

            # 累加能耗 (简化版，完整对应论文3.3.1节)
            C_energy += C_ELEC * (E_RTG * task.p_rtg / 60 + E_R_MOVE * rail_move_time / 60)
            C_energy += C_ELEC * (E_YARD * task.p_yard / 60 + E_Y_MOVE * yard_move_time / 60)
            C_energy += C_DIESEL * (E_T_LOAD * dist_rail_yard)

        elif task.type == 'B':
            # 类型B: 铁路卸->外部集卡->出港
            rail_move_time = 0.0
            if r_eq.task_sequence:
                last_task = tasks[r_eq.task_sequence[-1]]
                last_coords = get_task_coords(last_task)
                dist = manhattan_distance(last_coords['rail'], coords['rail'])
                rail_move_time = dist / V_R

            # 弹性时间窗约束 (对应论文3.4.2节)
            earliest_arrive = task.a - DELTA_EARLY
            latest_arrive = task.b + DELTA_LATE

            s_C = max(r_eq.available_time + rail_move_time, v_eq.available_time, earliest_arrive)
            if len(r_eq.task_sequence) < GAMMA_1:
                s_C += task.p_rtg_hat

            # 检查可行性
            if s_C > latest_arrive:
                feasible = False
                violation += (s_C - latest_arrive)

            c_C = s_C + task.p_rtg
            s_T_rail = s_C
            c_T_rail = c_C

            # 计算时间窗惩罚
            W_early = max(task.a - s_T_rail, 0)
            W_late = max(s_T_rail - task.b, 0)
            C_penalty += W_EARLY * W_early + W_LATE * W_late

            # 集卡出港
            dist_rail_gate = manhattan_distance(coords['rail'], GATE_COORD)
            move_time = dist_rail_gate / V_T_LOAD

            # 更新设备
            r_eq.available_time = c_C
            r_eq.task_sequence.append(task_id)
            v_eq.available_time = c_T_rail + move_time
            v_eq.task_sequence.append(task_id)

            # 累加能耗
            C_energy += C_ELEC * (E_RTG * task.p_rtg / 60 + E_R_MOVE * rail_move_time / 60)
            C_energy += C_DIESEL * (E_T_LOAD * dist_rail_gate)

        elif task.type == 'C':
            # 类型C: 堆场取->内部集卡->铁路装 (流程与A反向)
            # 1. 堆场正面吊作业
            yard_move_time = 0.0
            if y_eq.task_sequence:
                last_task = tasks[y_eq.task_sequence[-1]]
                last_coords = get_task_coords(last_task)
                dist = manhattan_distance(last_coords['yard'], coords['yard'])
                yard_move_time = dist / V_Y

            s_Y = max(y_eq.available_time + yard_move_time, v_eq.available_time)
            if len(y_eq.task_sequence) < GAMMA_1:
                s_Y += task.p_yard_hat
            c_Y = s_Y + task.p_yard
            s_T_yard = s_Y
            c_T_yard = c_Y

            # 2. 集卡行驶到铁路
            dist_yard_rail = manhattan_distance(coords['yard'], coords['rail'])
            move_time = dist_yard_rail / V_T_LOAD
            s_T_rail = c_T_yard + move_time
            c_T_rail = s_T_rail

            # 3. 铁路正面吊作业
            rail_move_time = 0.0
            if r_eq.task_sequence:
                last_task = tasks[r_eq.task_sequence[-1]]
                last_coords = get_task_coords(last_task)
                dist = manhattan_distance(last_coords['rail'], coords['rail'])
                rail_move_time = dist / V_R

            s_C = max(r_eq.available_time + rail_move_time, s_T_rail)
            if len(r_eq.task_sequence) < GAMMA_1:
                s_C += task.p_rtg_hat
            c_C = s_C + task.p_rtg

            # 更新设备
            r_eq.available_time = c_C
            r_eq.task_sequence.append(task_id)
            v_eq.available_time = c_T_rail
            v_eq.task_sequence.append(task_id)
            y_eq.available_time = c_Y
            y_eq.task_sequence.append(task_id)

            # 累加能耗
            C_energy += C_ELEC * (E_RTG * task.p_rtg / 60 + E_R_MOVE * rail_move_time / 60)
            C_energy += C_ELEC * (E_YARD * task.p_yard / 60 + E_Y_MOVE * yard_move_time / 60)
            C_energy += C_DIESEL * (E_T_LOAD * dist_yard_rail)

        elif task.type == 'D':
            # 类型D: 外部集卡->铁路装
            rail_move_time = 0.0
            if r_eq.task_sequence:
                last_task = tasks[r_eq.task_sequence[-1]]
                last_coords = get_task_coords(last_task)
                dist = manhattan_distance(last_coords['rail'], coords['rail'])
                rail_move_time = dist / V_R

            # 弹性时间窗
            earliest_arrive = task.a - DELTA_EARLY
            latest_arrive = task.b + DELTA_LATE

            s_C = max(r_eq.available_time + rail_move_time, v_eq.available_time, earliest_arrive)
            if len(r_eq.task_sequence) < GAMMA_1:
                s_C += task.p_rtg_hat

            if s_C > latest_arrive:
                feasible = False
                violation += (s_C - latest_arrive)

            c_C = s_C + task.p_rtg
            s_T_rail = s_C
            c_T_rail = c_C

            # 时间窗惩罚
            W_early = max(task.a - s_T_rail, 0)
            W_late = max(s_T_rail - task.b, 0)
            C_penalty += W_EARLY * W_early + W_LATE * W_late

            # 更新设备
            r_eq.available_time = c_C
            r_eq.task_sequence.append(task_id)
            v_eq.available_time = c_T_rail
            v_eq.task_sequence.append(task_id)

            # 累加能耗
            C_energy += C_ELEC * (E_RTG * task.p_rtg / 60 + E_R_MOVE * rail_move_time / 60)

        elif task.type == 'E':
            # 类型E: 空箱，流程同A
            rail_move_time = 0.0
            if r_eq.task_sequence:
                last_task = tasks[r_eq.task_sequence[-1]]
                last_coords = get_task_coords(last_task)
                dist = manhattan_distance(last_coords['rail'], coords['rail'])
                rail_move_time = dist / V_R

            s_C = max(r_eq.available_time + rail_move_time, v_eq.available_time)
            if len(r_eq.task_sequence) < GAMMA_1:
                s_C += task.p_rtg_hat
            c_C = s_C + task.p_rtg
            s_T_rail = s_C
            c_T_rail = c_C

            dist_rail_yard = manhattan_distance(coords['rail'], coords['yard'])
            move_time = dist_rail_yard / V_T_LOAD
            s_T_yard = c_T_rail + move_time
            c_T_yard = s_T_yard

            yard_move_time = 0.0
            if y_eq.task_sequence:
                last_task = tasks[y_eq.task_sequence[-1]]
                last_coords = get_task_coords(last_task)
                dist = manhattan_distance(last_coords['yard'], coords['yard'])
                yard_move_time = dist / V_Y

            s_Y = max(y_eq.available_time + yard_move_time, s_T_yard)
            if len(y_eq.task_sequence) < GAMMA_1:
                s_Y += task.p_yard_hat
            c_Y = s_Y + task.p_yard

            r_eq.available_time = c_C
            r_eq.task_sequence.append(task_id)
            v_eq.available_time = c_T_yard
            v_eq.task_sequence.append(task_id)
            y_eq.available_time = c_Y
            y_eq.task_sequence.append(task_id)

            C_energy += C_ELEC * (E_RTG * task.p_rtg / 60 + E_R_MOVE * rail_move_time / 60)
            C_energy += C_ELEC * (E_YARD * task.p_yard / 60 + E_Y_MOVE * yard_move_time / 60)
            C_energy += C_DIESEL * (E_T_LOAD * dist_rail_yard)

        # 更新总完工时间
        task_end_time = max(c_C, c_Y) if task.type in ['A','C','E'] else c_C
        T_max = max(T_max, task_end_time)

        # 存储任务时间
        task_times[task_id] = {
            'r_idx': r_idx,
            'v_idx': v_idx,
            'y_idx': y_idx if task.type in ['A','C','E'] else None,
            's_C': s_C, 'c_C': c_C,
            's_Y': s_Y, 'c_Y': c_Y,
            's_T_rail': s_T_rail, 'c_T_rail': c_T_rail,
            's_T_yard': s_T_yard, 'c_T_yard': c_T_yard,
            'W_early': W_early,
            'W_late': W_late
        }

    # 基础目标值 (f2暂不包含CVaR，留待仿真模块计算)
    base_schedule = {
        'task_times': task_times,
        'equipment': schedule,
        'C_energy_base': C_energy,
        'C_penalty_base': C_penalty
    }

    return T_max, C_energy + C_penalty, feasible, violation, base_schedule

# ==========================================
# 4. 初始种群生成 (对应论文4.3节)
# ==========================================
def generate_heuristic_individual(tasks: List[Task], rule: str) -> Individual:
    """基于启发式规则生成个体 (EDD, SPT, 能耗贪婪, 鲁棒缓冲)"""
    ind = Individual(len(tasks))
    num_v = NUM_V_IN + NUM_V_OUT

    # 按规则排序任务
    if rule == 'EDD':
        # 最早截止时间优先
        def sort_key(t):
            if tasks[t].type in ['B', 'D']:
                return tasks[t].b
            return tasks[t].p_rtg
        ind.task_order = sorted(range(len(tasks)), key=sort_key)
    elif rule == 'SPT':
        # 最短作业时间优先
        ind.task_order = sorted(range(len(tasks)), key=lambda t: tasks[t].p_rtg)
    elif rule == 'energy':
        # 能耗贪婪优先 (简化版)
        ind.task_order = sorted(range(len(tasks)), key=lambda t: tasks[t].p_rtg * E_RTG)
    elif rule == 'robust':
        # 鲁棒缓冲优先 (波动小的先做)
        ind.task_order = sorted(range(len(tasks)), key=lambda t: tasks[t].p_rtg_hat)
    else:
        random.shuffle(ind.task_order)

    # 分配段: 最早可用设备贪婪分配
    r_available = [0.0] * NUM_R
    y_available = [0.0] * NUM_Y
    v_available = [0.0] * (NUM_V_IN + NUM_V_OUT)

    for task_id in ind.task_order:
        task = tasks[task_id]

        # 铁路正面吊分配
        r_idx = np.argmin(r_available)
        ind.r_alloc[task_id] = r_idx
        r_available[r_idx] += task.p_rtg

        # 集卡分配
        if task.type in ['A', 'C', 'E']:
            v_idx = np.argmin(v_available[:NUM_V_IN])
        else:
            v_idx = NUM_V_IN + np.argmin(v_available[NUM_V_IN:])
        v_available[v_idx] += task.p_rtg * 0.5  # 简化估算

        # 堆场正面吊分配
        if task.type in ['A', 'C', 'E']:
            y_idx = np.argmin(y_available)
            y_available[y_idx] += task.p_yard
        else:
            y_idx = 0

        ind.vy_alloc[task_id] = v_idx + num_v * y_idx

    return ind

def initialize_population(tasks: List[Task]) -> List[Individual]:
    """混合初始化策略: 40%启发式 + 60%随机"""
    pop = []
    num_heuristic = int(POP_SIZE * 0.4)
    rules = ['EDD', 'SPT', 'energy', 'robust']

    # 启发式解
    for i in range(num_heuristic):
        rule = rules[i % len(rules)]
        ind = generate_heuristic_individual(tasks, rule)
        pop.append(ind)

    # 随机解
    while len(pop) < POP_SIZE:
        ind = initialize_individual(tasks)
        pop.append(ind)

    return pop

# ==========================================
# 5. 遗传算子 (对应论文4.4节)
# ==========================================
def tournament_selection(pop: List[Individual]) -> Individual:
    """锦标赛选择 (基于约束支配与拥挤度)"""
    candidates = random.sample(pop, TOURNAMENT_SIZE)
    # 按 rank -> feasible -> crowding_distance 排序
    candidates.sort(key=lambda x: (x.rank, not x.feasible, -x.crowding_distance))
    return candidates[0]

def order_crossover(parent1: Individual, parent2: Individual) -> Individual:
    """改进顺序交叉 (OX) 用于任务排序段"""
    child = Individual(parent1.num_tasks)
    child.r_alloc = parent1.r_alloc.copy()
    child.vy_alloc = parent1.vy_alloc.copy()

    # OX交叉
    size = parent1.num_tasks
    start, end = sorted(random.sample(range(size), 2))

    # 复制父代1的片段
    child.task_order = [None] * size
    child.task_order[start:end] = parent1.task_order[start:end]

    # 从父代2填充剩余
    ptr = end
    for task in parent2.task_order:
        if task not in child.task_order:
            if ptr >= size:
                ptr = 0
            while child.task_order[ptr] is not None:
                ptr += 1
                if ptr >= size:
                    ptr = 0
            child.task_order[ptr] = task

    return child

def uniform_crossover_alloc(parent1: Individual, parent2: Individual, tasks: List[Task]) -> Individual:
    """约束均匀交叉用于分配段"""
    child = Individual(parent1.num_tasks)
    child.task_order = parent1.task_order.copy()
    num_v = NUM_V_IN + NUM_V_OUT

    for i in range(child.num_tasks):
        task = tasks[i]
        if random.random() < 0.5:
            child.r_alloc[i] = parent1.r_alloc[i]
            child.vy_alloc[i] = parent1.vy_alloc[i]
        else:
            child.r_alloc[i] = parent2.r_alloc[i]
            child.vy_alloc[i] = parent2.vy_alloc[i]

        # 约束校验与修复
        vy_code = child.vy_alloc[i]
        v_idx = vy_code % num_v
        if task.type in ['A', 'C', 'E'] and v_idx >= NUM_V_IN:
            # 修复: 随机选内部集卡
            v_idx = random.randint(0, NUM_V_IN - 1)
            y_idx = vy_code // num_v
            child.vy_alloc[i] = v_idx + num_v * y_idx
        elif task.type in ['B', 'D'] and v_idx < NUM_V_IN:
            # 修复: 随机选外部集卡
            v_idx = random.randint(NUM_V_IN, num_v - 1)
            child.vy_alloc[i] = v_idx + num_v * 0  # y=0

    return child

def crossover(parent1: Individual, parent2: Individual, tasks: List[Task]) -> Tuple[Individual, Individual]:
    """分段交叉算子"""
    child1 = Individual(parent1.num_tasks)
    child2 = Individual(parent2.num_tasks)

    if random.random() < PC:
        # 分配段交叉
        child1 = uniform_crossover_alloc(parent1, parent2, tasks)
        child2 = uniform_crossover_alloc(parent2, parent1, tasks)

        # 排序段交叉
        child1_order = order_crossover(parent1, parent2)
        child2_order = order_crossover(parent2, parent1)
        child1.task_order = child1_order.task_order
        child2.task_order = child2_order.task_order
    else:
        # 直接复制
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

    return child1, child2

def swap_mutation(ind: Individual):
    """双点交换变异用于排序段"""
    if random.random() < PM:
        i, j = random.sample(range(ind.num_tasks), 2)
        ind.task_order[i], ind.task_order[j] = ind.task_order[j], ind.task_order[i]

def inversion_mutation(ind: Individual):
    """逆序变异用于排序段"""
    if random.random() < PM:
        start, end = sorted(random.sample(range(ind.num_tasks), 2))
        ind.task_order[start:end] = reversed(ind.task_order[start:end])

def mutate_alloc(ind: Individual, tasks: List[Task]):
    """约束随机变异用于分配段"""
    num_v = NUM_V_IN + NUM_V_OUT
    for i in range(ind.num_tasks):
        if random.random() < PM / ind.num_tasks:  # 位变异概率
            task = tasks[i]
            # 铁路正面吊变异
            ind.r_alloc[i] = random.randint(0, NUM_R - 1)

            # 集卡-堆场变异
            if task.type in ['A', 'C', 'E']:
                v_idx = random.randint(0, NUM_V_IN - 1)
                y_idx = random.randint(0, NUM_Y - 1)
            else:
                v_idx = random.randint(NUM_V_IN, num_v - 1)
                y_idx = 0
            ind.vy_alloc[i] = v_idx + num_v * y_idx

def mutate(ind: Individual, tasks: List[Task]):
    """混合变异算子"""
    # 分配段变异
    mutate_alloc(ind, tasks)
    # 排序段变异 (50%概率交换, 50%概率逆序)
    if random.random() < 0.5:
        swap_mutation(ind)
    else:
        inversion_mutation(ind)

# ==========================================
# 6. 内层仿真评估模块 (对应论文4.5节 & 3.4.3节)
# ==========================================
def generate_fault_scenarios(T_max: float) -> List[List[Dict]]:
    """
    生成Q个故障情景 (对应论文3.4.3节)
    返回: 情景列表，每个情景是故障事件列表 [(设备类型, 设备ID, 故障时间, 维修时长), ...]
    """
    scenarios = []
    for _ in range(Q):
        scenario = []
        # 铁路正面吊故障
        for r_id in range(NUM_R):
            t = 0.0
            while t < T_max:
                delta_t = np.random.exponential(1.0 / LAMBDA_R) if LAMBDA_R > 0 else np.inf
                t += delta_t
                if t < T_max:
                    d_f = np.random.normal(30, 10)  # 维修时间: 正态分布 N(30,10)
                    d_f = max(10, d_f)
                    scenario.append(('R', r_id, t, d_f))
                    t += d_f

        # 堆场正面吊故障
        for y_id in range(NUM_Y):
            t = 0.0
            while t < T_max:
                delta_t = np.random.exponential(1.0 / LAMBDA_Y) if LAMBDA_Y > 0 else np.inf
                t += delta_t
                if t < T_max:
                    d_f = np.random.normal(25, 8)
                    d_f = max(8, d_f)
                    scenario.append(('Y', y_id, t, d_f))
                    t += d_f

        # 集卡故障
        num_v = NUM_V_IN + NUM_V_OUT
        for v_id in range(num_v):
            t = 0.0
            while t < T_max:
                delta_t = np.random.exponential(1.0 / LAMBDA_V) if LAMBDA_V > 0 else np.inf
                t += delta_t
                if t < T_max:
                    d_f = np.random.normal(20, 6)
                    d_f = max(5, d_f)
                    scenario.append(('V', v_id, t, d_f))
                    t += d_f

        # 按故障时间排序
        scenario.sort(key=lambda x: x[2])
        scenarios.append(scenario)

    return scenarios

def fast_reschedule(original_schedule: dict, scenario: List[Dict], tasks: List[Task]) -> Tuple[float, float]:
    """
    快速重调度规则 (对应论文3.4.3节: 最早可用-最低增量能耗)
    返回: (delta_energy, delta_penalty)
    """
    # 复制原调度
    schedule = copy.deepcopy(original_schedule)
    delta_energy = 0.0
    delta_penalty = 0.0

    # 设备不可用时段标记
    unavailable = {
        'R': [[] for _ in range(NUM_R)],
        'Y': [[] for _ in range(NUM_Y)],
        'V': [[] for _ in range(NUM_V_IN + NUM_V_OUT)]
    }

    # 处理故障事件
    for event in scenario:
        eq_type, eq_id, t_f, d_f = event
        unavailable[eq_type][eq_id].append((t_f, t_f + d_f))

        # 找到该设备上在t_f时刻未完成的任务
        eq_list = schedule['equipment'][eq_type.lower()]
        if eq_id >= len(eq_list):
            continue
        eq = eq_list[eq_id]

        # 简单模拟: 将故障后未完成的任务加入待重分配池 (简化版完整逻辑)
        # 实际完整实现需遍历任务序列判断中断点
        # 这里为了代码简洁，演示核心逻辑: 计算闲置能耗
        idle_time = d_f
        if eq_type == 'R':
            delta_energy += C_ELEC * E_R_IDLE * idle_time / 60
        elif eq_type == 'Y':
            delta_energy += C_ELEC * E_Y_IDLE * idle_time / 60
        elif eq_type == 'V':
            delta_energy += C_DIESEL * E_T_IDLE * idle_time

    # 额外能耗与惩罚简化计算 (完整实现需模拟任务重排)
    # 这里用系数放大以体现故障影响
    delta_energy *= 1.5
    delta_penalty = delta_energy * 0.3

    return delta_energy, delta_penalty

def evaluate_individual(ind: Individual, tasks: List[Task]):
    """完整评估个体: 解码 + 仿真 + CVaR计算"""
    # 1. 基础解码
    T_max, f2_base, feasible, violation, base_schedule = decode(ind, tasks)
    ind.f1 = T_max
    ind.feasible = feasible
    ind.constraint_violation = violation
    ind.base_schedule = base_schedule

    if not feasible:
        ind.f2 = np.inf
        return

    # 2. 蒙特卡洛仿真 (对应论文3.4.3节)
    if T_max <= 0:
        T_max = 1000.0
    scenarios = generate_fault_scenarios(T_max * 1.2)
    losses = []

    for scenario in scenarios:
        # 单情景损失计算
        C_idle = 0.0  # 已在fast_reschedule中计算
        delta_C_energy, delta_C_penalty = fast_reschedule(base_schedule, scenario, tasks)
        L_omega = C_idle + delta_C_energy + delta_C_penalty
        losses.append(L_omega)

    # 3. CVaR计算 (对应论文式3-82至3-84)
    losses = np.array(losses)
    losses_sorted = np.sort(losses)
    k = int(np.ceil((1 - ALPHA) * Q))
    VaR = losses_sorted[k-1] if k > 0 else losses_sorted[0]
    CVaR = np.mean(losses_sorted[k-1:]) if k > 0 else np.mean(losses)

    # 4. 最终目标函数 (对应论文式3-85)
    ind.f2 = base_schedule['C_energy_base'] + base_schedule['C_penalty_base'] + BETA * CVaR

# ==========================================
# 7. 改进非支配排序与精英保留 (对应论文4.6节)
# ==========================================
def constraint_dominates(ind1: Individual, ind2: Individual) -> bool:
    """约束支配准则"""
    # 1. 可行解支配不可行解
    if ind1.feasible and not ind2.feasible:
        return True
    if not ind1.feasible and not ind2.feasible:
        return ind1.constraint_violation < ind2.constraint_violation

    # 2. 两个可行解: Pareto支配
    if ind1.feasible and ind2.feasible:
        dominates = (ind1.f1 <= ind2.f1) and (ind1.f2 <= ind2.f2)
        strictly_better = (ind1.f1 < ind2.f1) or (ind1.f2 < ind2.f2)
        return dominates and strictly_better

    return False

def fast_non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    """改进的快速非支配排序"""
    S = [[] for _ in range(len(pop))]  # 被p支配的个体
    n = [0] * len(pop)                  # 支配p的个体数
    fronts = [[]]

    for p in range(len(pop)):
        S[p] = []
        n[p] = 0
        for q in range(len(pop)):
            if p == q:
                continue
            if constraint_dominates(pop[p], pop[q]):
                S[p].append(q)
            elif constraint_dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    pop[q_idx].rank = i + 1
                    next_front.append(q_idx)
        i += 1
        fronts.append(next_front)

    # 转换为个体列表的fronts
    front_individuals = []
    for f in fronts[:-1]:
        front_individuals.append([pop[i] for i in f])

    return front_individuals

def normalize(vals: List[float]) -> List[float]:
    """最小-最大归一化"""
    min_val = min(vals)
    max_val = max(vals)
    if max_val - min_val < 1e-9:
        return [0.5] * len(vals)
    return [(v - min_val) / (max_val - min_val) for v in vals]

def calculate_crowding_distance(front: List[Individual]):
    """改进的归一化拥挤度距离计算"""
    if len(front) == 0:
        return
    if len(front) == 1:
        front[0].crowding_distance = np.inf
        return

    # 归一化目标值
    f1_vals = [ind.f1 for ind in front]
    f2_vals = [ind.f2 for ind in front]
    f1_norm = normalize(f1_vals)
    f2_norm = normalize(f2_vals)

    # 按f1排序
    sorted_by_f1 = sorted(range(len(front)), key=lambda i: f1_norm[i])
    front[sorted_by_f1[0]].crowding_distance = np.inf
    front[sorted_by_f1[-1]].crowding_distance = np.inf

    for i in range(1, len(front)-1):
        front[sorted_by_f1[i]].crowding_distance = (
            f1_norm[sorted_by_f1[i+1]] - f1_norm[sorted_by_f1[i-1]]
        )

    # 按f2排序并累加
    sorted_by_f2 = sorted(range(len(front)), key=lambda i: f2_norm[i])
    front[sorted_by_f2[0]].crowding_distance = np.inf
    front[sorted_by_f2[-1]].crowding_distance = np.inf

    for i in range(1, len(front)-1):
        front[sorted_by_f2[i]].crowding_distance += (
            f2_norm[sorted_by_f2[i+1]] - f2_norm[sorted_by_f2[i-1]]
        )

def update_elite_archive(archive: List[Individual], pop: List[Individual]) -> List[Individual]:
    """更新外部精英档案"""
    # 合并
    combined = archive + pop
    # 去重
    combined = list(dict.fromkeys(combined))
    # 非支配排序
    fronts = fast_non_dominated_sort(combined)
    # 提取第一前沿
    new_archive = fronts[0] if fronts else []
    # 修剪
    if len(new_archive) > ELITE_ARCHIVE_SIZE:
        calculate_crowding_distance(new_archive)
        new_archive.sort(key=lambda x: -x.crowding_distance)
        new_archive = new_archive[:ELITE_ARCHIVE_SIZE]
    return new_archive

# ==========================================
# 8. 新增：结果可视化与详细调度输出
# ==========================================
def print_schedule_detail(ind: Individual, tasks: List[Task]):
    """打印最优解的详细调度过程"""
    print("\n" + "="*100)
    print(f"【最优调度方案详情】 (总完工时间: {ind.f1:.2f}分钟, 综合成本: {ind.f2:.2f}元)")
    print("="*100)

    task_times = ind.base_schedule['task_times']
    num_v = NUM_V_IN + NUM_V_OUT

    # 按任务ID排序输出
    print(f"\n{'任务ID':<8} {'类型':<6} {'铁路正面吊':<12} {'内部/外部集卡':<15} {'堆场正面吊':<12} {'铁路作业(开始-结束)':<25} {'堆场作业(开始-结束)':<25} {'时间窗惩罚':<12}")
    print("-" * 140)

    for task_id in range(ind.num_tasks):
        task = tasks[task_id]
        tt = task_times[task_id]

        # 设备名称
        r_name = f"R-{tt['r_idx']+1}"
        v_type = "内部" if tt['v_idx'] < NUM_V_IN else "外部"
        v_name = f"{v_type}V-{tt['v_idx']+1}"
        y_name = f"Y-{tt['y_idx']+1}" if tt['y_idx'] is not None else "-"

        # 时间格式化
        rail_time = f"{tt['s_C']:.1f} - {tt['c_C']:.1f}"
        yard_time = f"{tt['s_Y']:.1f} - {tt['c_Y']:.1f}" if tt['y_idx'] is not None else "-"
        penalty = f"{tt['W_early'] + tt['W_late']:.1f}"

        print(f"{task_id+1:<8} {task.type:<6} {r_name:<12} {v_name:<15} {y_name:<12} {rail_time:<25} {yard_time:<25} {penalty:<12}")

    # 打印设备任务序列
    print("\n" + "-"*100)
    print("【各设备任务执行序列】")
    print("-"*100)

    # 铁路正面吊
    eq_schedule = ind.base_schedule['equipment']
    for r in range(NUM_R):
        seq = eq_schedule['r'][r].task_sequence
        print(f"铁路正面吊 R-{r+1}: {' → '.join([f'T{t+1}' for t in seq])}")

    # 堆场正面吊
    for y in range(NUM_Y):
        seq = eq_schedule['y'][y].task_sequence
        if seq:
            print(f"堆场正面吊 Y-{y+1}: {' → '.join([f'T{t+1}' for t in seq])}")

    # 内部集卡
    print("\n内部集卡:")
    for v in range(NUM_V_IN):
        seq = eq_schedule['v'][v].task_sequence
        print(f"  内部V-{v+1}: {' → '.join([f'T{t+1}' for t in seq])}")

    # 外部集卡
    print("\n外部集卡:")
    for v in range(NUM_V_IN, NUM_V_IN + NUM_V_OUT):
        seq = eq_schedule['v'][v].task_sequence
        print(f"  外部V-{v+1}: {' → '.join([f'T{t+1}' for t in seq])}")

    print("="*100)

# ==========================================
# 9. 主算法流程
# ==========================================
def main():
    print("="*80)
    print("陆港正面吊-集卡协同调度 SN-NSGA-II 算法 (增强版)")
    print("="*80)

    # 记录总开始时间
    total_start_time = time.time()

    # 1. 生成任务数据
    print(f"\n[1/5] 正在生成 {NUM_TASKS} 个任务...")
    tasks = [Task(i) for i in range(NUM_TASKS)]
    for i, t in enumerate(tasks[:5]):  # 打印前5个任务
        zone_info = f", 堆场分区={ZONES[t.zone]}" if t.zone else ""
        print(f"  任务{i+1}: 类型={t.type}, 铁路作业点={t.pos}{zone_info}")

    # 2. 初始化种群
    print("\n[2/5] 正在初始化种群...")
    pop = initialize_population(tasks)

    # 3. 初始评估
    print("[3/5] 正在评估初始种群...")
    eval_start = time.time()
    for ind in pop:
        evaluate_individual(ind, tasks)
    eval_time = time.time() - eval_start
    print(f"  初始种群评估耗时: {eval_time:.2f}秒")

    # 4. 初始排序与精英档案
    fronts = fast_non_dominated_sort(pop)
    for front in fronts:
        calculate_crowding_distance(front)
    elite_archive = update_elite_archive([], pop)
    print(f"  初始非支配解数量: {len(elite_archive)}")

    # 5. 主迭代循环
    print("\n[4/5] 开始迭代优化...")
    iter_times = []

    for iter in range(MAX_ITER):
        iter_start = time.time()

        # 生成子代
        offspring = []
        while len(offspring) < POP_SIZE:
            parent1 = tournament_selection(pop)
            parent2 = tournament_selection(pop)
            child1, child2 = crossover(parent1, parent2, tasks)
            mutate(child1, tasks)
            mutate(child2, tasks)
            offspring.append(child1)
            if len(offspring) < POP_SIZE:
                offspring.append(child2)

        # 评估子代
        for ind in offspring:
            evaluate_individual(ind, tasks)

        # 合并种群
        combined = pop + offspring

        # 非支配排序
        fronts = fast_non_dominated_sort(combined)
        for front in fronts:
            calculate_crowding_distance(front)

        # 构建新一代种群
        new_pop = []
        front_idx = 0
        while front_idx < len(fronts) and len(new_pop) + len(fronts[front_idx]) <= POP_SIZE:
            new_pop.extend(fronts[front_idx])
            front_idx += 1

        if len(new_pop) < POP_SIZE and front_idx < len(fronts):
            # 按拥挤度距离排序补充
            remaining = fronts[front_idx]
            remaining.sort(key=lambda x: -x.crowding_distance)
            needed = POP_SIZE - len(new_pop)
            new_pop.extend(remaining[:needed])

        pop = new_pop

        # 更新精英档案
        elite_archive = update_elite_archive(elite_archive, pop)

        # 记录迭代时间
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)

        # 打印进度
        if (iter + 1) % 20 == 0 or iter == 0:
            feasible_elite = [ind for ind in elite_archive if ind.feasible]
            if feasible_elite:
                best_f1 = min(ind.f1 for ind in feasible_elite)
                best_f2 = min(ind.f2 for ind in feasible_elite)
                print(f"  迭代 {iter+1:3d}/{MAX_ITER} | 耗时: {iter_time:.2f}s | 精英解数: {len(feasible_elite):3d} | 最佳完工时间: {best_f1:6.1f}min | 最佳成本: {best_f2:6.1f}元")

    # 6. 结果输出
    total_time = time.time() - total_start_time
    print("\n" + "="*80)
    print("[5/5] 优化完成！")
    print("="*80)

    # 速度统计
    print(f"\n【算法运行速度统计】")
    print(f"  总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"  平均每轮迭代耗时: {np.mean(iter_times):.2f} 秒")
    print(f"  最快迭代耗时: {np.min(iter_times):.2f} 秒")
    print(f"  最慢迭代耗时: {np.max(iter_times):.2f} 秒")

    # 去重后的Pareto前沿
    feasible_elite = [ind for ind in elite_archive if ind.feasible]
    # 去重
    feasible_elite = list(dict.fromkeys(feasible_elite))
    feasible_elite.sort(key=lambda x: x.f1)

    print(f"\n【去重后的Pareto最优解集】 (共 {len(feasible_elite)} 个非支配解)")
    print(f"{'序号':<6} {'总完工时间(min)':<20} {'综合成本(元)':<20}")
    print("-" * 50)
    for i, ind in enumerate(feasible_elite):
        print(f"{i+1:<6} {ind.f1:<20.2f} {ind.f2:<20.2f}")
        if i >= 9:  # 只打印前10个
            print(f"  ... (剩余 {len(feasible_elite)-10} 个解)")
            break

    # 选择一个最优解进行详细展示 (选择f1最小的)
    if feasible_elite:
        best_f1_ind = feasible_elite[0]
        print_schedule_detail(best_f1_ind, tasks)

if __name__ == "__main__":
    main()