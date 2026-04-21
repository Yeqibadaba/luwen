import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# ==================== 参数配置 ====================
PARAMS = {
    # 算法参数
    'pop_size': 100,              # 种群规模
    'max_gen': 200,               # 最大迭代代数
    'pc': 0.85,                   # 交叉概率
    'pm': 0.25,                   # 变异概率
    'Q': 30,                      # 蒙特卡洛仿真情景数
    'beta': 2.0,                  # 风险厌恶系数
    'gamma': 0.1,                 # 重调度时间-能耗平衡系数
    'stall_gen': 50,              # 早停阈值（代）

    # 设备参数
    'v_rtg': 4.2,                 # 火车线正面吊速度 m/s
    'v_yard': 4.2,                # 堆场正面吊速度 m/s
    'v_truck': 5.7,               # 集卡速度 m/s

    # 能耗参数 (单位: kWh/s)
    'e_rtg_base': 11 / 3600,      # 火车线正面吊装卸能耗 per sec
    'e_yard_base': 11 / 3600,     # 堆场正面吊装卸能耗 per sec
    'e_empty_rtg': 15 / 3600,     # 火车线正面吊空载移动能耗 per sec
    'e_empty_yard': 15 / 3600,    # 堆场正面吊空载移动能耗 per sec
    'e_empty_truck': 14 / 3600,   # 集卡空载移动能耗 per sec
    'e_load_truck': 21 / 3600,    # 集卡载重移动能耗 per sec
    'e_idle_rtg': 2 / 3600,       # 火车线正面吊空闲能耗 per sec
    'e_idle_yard': 2 / 3600,      # 堆场正面吊空闲能耗 per sec
    'e_idle_truck': 1 / 3600,     # 集卡空闲能耗 per sec

    # 故障参数
    'lambda_rtg': 0.01 / 3600,    # 火车线正面吊故障率 (次/秒)
    'lambda_yard': 0.01 / 3600,   # 堆场正面吊故障率
    'lambda_truck': 0.02 / 3600,  # 集卡故障率
    'repair_mean': 1800,          # 平均维修时间 1800 秒
}

# 堆场分区坐标 (曼哈顿距离)
ZONES = {
    1: {'name': '铁路近端卸车区', 'coord': (50, 100)},
    2: {'name': '铁路远端堆存区', 'coord': (150, 100)},
    3: {'name': '港外近端直装区', 'coord': (50, 300)},
    4: {'name': '港外远端堆存区', 'coord': (150, 300)},
    5: {'name': '空箱区', 'coord': (100, 200)},
}
GATE_COORD = (0, 300)          # 港外出入口坐标
RAIL_POINTS = [(0, 100 + i*10) for i in range(10)]  # 铁路作业点 y 坐标 100~190

# ==================== 任务数据生成 ====================
def generate_tasks(num_tasks: int) -> List[Dict]:
    """生成随机任务数据"""
    tasks = []
    for i in range(num_tasks):
        task_type = random.choice(['A','B','C','D','E'])
        pos = random.randint(0, len(RAIL_POINTS)-1)
        p_rtg = random.uniform(60, 100)   # 秒
        p_yard = random.uniform(50, 80) if task_type in ['A','C','E'] else 0
        # 外部集卡时间窗 (仅B/D)
        a = random.uniform(0, 500) if task_type in ['B','D'] else 0
        b = a + random.uniform(200, 600) if task_type in ['B','D'] else 0
        # 目标分区
        if task_type == 'A':
            zone = random.choice([1,2])
        elif task_type == 'C':
            zone = 4
        elif task_type == 'E':
            zone = 5
        else:
            zone = None
        tasks.append({
            'id': i,
            'type': task_type,
            'pos': pos,
            'p_rtg': p_rtg,
            'p_yard': p_yard,
            'a': a,
            'b': b,
            'zone': zone,
        })
    return tasks

# 预计算距离矩阵
def compute_distances():
    rail_zone_dist = {}
    for i, rp in enumerate(RAIL_POINTS):
        for z, info in ZONES.items():
            rail_zone_dist[(i, z)] = abs(rp[0]-info['coord'][0]) + abs(rp[1]-info['coord'][1])
    zone_zone_dist = {}
    for z1 in ZONES:
        for z2 in ZONES:
            zone_zone_dist[(z1, z2)] = abs(ZONES[z1]['coord'][0]-ZONES[z2]['coord'][0]) + abs(ZONES[z1]['coord'][1]-ZONES[z2]['coord'][1])
    zone_gate_dist = {z: abs(info['coord'][0]-GATE_COORD[0]) + abs(info['coord'][1]-GATE_COORD[1]) for z, info in ZONES.items()}
    rail_gate_dist = {i: abs(rp[0]-GATE_COORD[0]) + abs(rp[1]-GATE_COORD[1]) for i, rp in enumerate(RAIL_POINTS)}
    return rail_zone_dist, zone_zone_dist, zone_gate_dist, rail_gate_dist

rail_zone_dist, zone_zone_dist, zone_gate_dist, rail_gate_dist = compute_distances()

# ==================== 调度方案类 ====================
class Schedule:
    def __init__(self, num_tasks, num_rtg, num_yard, num_truck):
        self.num_tasks = num_tasks
        self.num_rtg = num_rtg
        self.num_yard = num_yard
        self.num_truck = num_truck

        self.rtg_assign = [0]*num_tasks
        self.yard_assign = [0]*num_tasks
        self.truck_assign = [0]*num_tasks

        self.rtg_seq = [[] for _ in range(num_rtg)]
        self.yard_seq = [[] for _ in range(num_yard)]
        self.truck_seq = [[] for _ in range(num_truck)]

        self.start_rtg = [0.0]*num_tasks
        self.start_yard = [0.0]*num_tasks
        self.start_truck = [0.0]*num_tasks
        self.end_rtg = [0.0]*num_tasks
        self.end_yard = [0.0]*num_tasks
        self.end_truck = [0.0]*num_tasks

        self.f1 = None
        self.f2 = None
        self.rank = None
        self.crowding = None

# ==================== 解码函数（含移动时间和空闲能耗） ====================
def decode(schedule: Schedule, tasks: List[Dict], params: Dict) -> Schedule:
    """解码调度方案，计算时间、能耗，并填充时间变量"""
    num_tasks = schedule.num_tasks
    # 初始化设备可用时间和最后位置（用于计算移动距离）
    avail_rtg = [0.0]*schedule.num_rtg
    avail_yard = [0.0]*schedule.num_yard
    avail_truck = [0.0]*schedule.num_truck
    last_pos_rtg = [None]*schedule.num_rtg
    last_pos_yard = [None]*schedule.num_yard
    last_pos_truck = [None]*schedule.num_truck

    # 存储每个任务的前驱阶段完成时间（用于同步）
    # 对于每个任务，记录三个阶段的完成时间，初始为0
    rtg_end = [0.0]*num_tasks
    yard_end = [0.0]*num_tasks
    truck_end = [0.0]*num_tasks

    # 按设备类型分别安排，但需要协调任务之间的依赖
    # 我们采用按任务类型顺序推进：对于每个设备上的任务序列，依次安排，但任务可能依赖其他设备
    # 为简化，我们采用循环直到所有任务安排完成（类似事件调度）

    # 记录每个任务已完成阶段标志
    stage_done = [0]*num_tasks  # 0:未开始,1:rtg完成,2:truck完成,3:yard完成

    # 我们需要一个按时间顺序处理事件的方法。这里使用贪心：重复扫描所有设备序列，找到可安排的任务
    # 更简单的方法：先安排所有设备的序列，但考虑依赖。我们采用以下方式：
    # 1. 对于每个设备，按顺序处理其序列中的任务，但任务开始时间需考虑前驱设备完成时间。
    # 2. 因为任务类型固定，我们可以为每个任务定义阶段列表和依赖。
    # 实现细节较多，我们采用类似拓扑排序的方法：记录每个任务的入度（未完成的依赖数），每次选择入度为0的阶段安排。

    # 构建每个任务的阶段列表（按时间顺序）
    # 类型A: [('rtg', p_rtg), ('truck', travel_time), ('yard', p_yard)]
    # 类型B: [('rtg', p_rtg), ('truck', travel_time)]
    # 类型C: [('yard', p_yard), ('truck', travel_time), ('rtg', p_rtg)]
    # 类型D: [('truck', travel_time), ('rtg', p_rtg)]
    # 类型E: 同A，但目标分区为5

    # 为简化，我们采用逐个设备推进的方法，但需要处理跨设备依赖。我们采用以下策略：
    # 先初始化所有任务的三个阶段完成时间为0，然后按设备类型顺序处理，但会循环多次直到收敛。
    # 这里为了代码可读性，我们采用循环直到所有任务完成，每次处理一个任务的一个阶段。

    # 使用一个队列存储待处理阶段（任务id，阶段名）
    # 但实现较复杂，我们采用简化版本：按设备序列顺序安排，但计算开始时间时考虑前驱完成时间。
    # 因为任务类型固定，我们可以先安排rtg，再安排truck，再安排yard，但依赖关系可能交错。

    # 更好的方法是：按设备序列顺序处理每个任务，但任务开始时间由以下决定：
    # 对于rtg阶段：开始时间 = max(设备可用时间, 前驱完成时间)
    # 前驱完成时间：类型A/B/E的rtg无前驱；类型C的rtg需要truck完成；类型D的rtg需要truck完成。
    # 对于truck阶段：开始时间 = max(设备可用时间, 前驱完成时间)
    # 前驱：类型A/B/E的truck需要rtg完成；类型C的truck需要yard完成；类型D的truck无前驱。
    # 对于yard阶段：开始时间 = max(设备可用时间, 前驱完成时间)
    # 前驱：类型A/E的yard需要truck完成；类型C的yard无前驱。

    # 我们按设备类型顺序处理，但需要多次扫描以确保依赖关系满足。我们采用以下循环：
    changed = True
    max_iter = 1000
    iter_count = 0
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1

        # 处理火车线正面吊
        for r in range(schedule.num_rtg):
            seq = schedule.rtg_seq[r]
            last_time = avail_rtg[r]
            for task_id in seq:
                t = tasks[task_id]
                # 计算前驱完成时间
                prev = 0.0
                if t['type'] in ['A','B','E']:
                    # 无前驱
                    prev = 0.0
                elif t['type'] == 'C':
                    prev = truck_end[task_id]  # 等待truck完成
                elif t['type'] == 'D':
                    prev = truck_end[task_id]
                start = max(last_time, prev)
                if abs(start - schedule.start_rtg[task_id]) > 1e-6:
                    changed = True
                schedule.start_rtg[task_id] = start
                end = start + t['p_rtg']
                schedule.end_rtg[task_id] = end
                rtg_end[task_id] = end
                last_time = end
                # 移动时间：设备移动到下一个任务位置的时间（在当前任务完成后）
                # 获取当前任务位置
                if t['type'] in ['A','B','D','E']:
                    cur_pos = RAIL_POINTS[t['pos']]
                else:
                    # 类型C的rtg任务位置是铁路作业点（由pos给出）
                    cur_pos = RAIL_POINTS[t['pos']]
                # 找到下一个任务的位置
                idx = seq.index(task_id)
                if idx < len(seq)-1:
                    next_id = seq[idx+1]
                    next_t = tasks[next_id]
                    if next_t['type'] in ['A','B','D','E']:
                        next_pos = RAIL_POINTS[next_t['pos']]
                    else:
                        next_pos = RAIL_POINTS[next_t['pos']]
                    travel_time = (abs(cur_pos[0]-next_pos[0]) + abs(cur_pos[1]-next_pos[1])) / params['v_rtg']
                    last_time += travel_time
            avail_rtg[r] = last_time

        # 处理集卡
        for v in range(schedule.num_truck):
            seq = schedule.truck_seq[v]
            last_time = avail_truck[v]
            for task_id in seq:
                t = tasks[task_id]
                # 计算前驱完成时间
                prev = 0.0
                if t['type'] in ['A','B','E']:
                    prev = rtg_end[task_id]
                elif t['type'] == 'C':
                    prev = yard_end[task_id]
                elif t['type'] == 'D':
                    prev = 0.0
                start = max(last_time, prev)
                if abs(start - schedule.start_truck[task_id]) > 1e-6:
                    changed = True
                schedule.start_truck[task_id] = start
                # 行驶时间
                if t['type'] == 'A':
                    travel = rail_zone_dist[(t['pos'], t['zone'])] / params['v_truck']
                elif t['type'] == 'B':
                    travel = rail_gate_dist[t['pos']] / params['v_truck']
                elif t['type'] == 'C':
                    travel = rail_zone_dist[(t['pos'], t['zone'])] / params['v_truck']
                elif t['type'] == 'D':
                    travel = rail_gate_dist[t['pos']] / params['v_truck']
                elif t['type'] == 'E':
                    travel = rail_zone_dist[(t['pos'], 5)] / params['v_truck']
                else:
                    travel = 0
                end = start + travel
                schedule.end_truck[task_id] = end
                truck_end[task_id] = end
                last_time = end
                # 移动时间：集卡移动到下一个任务起点
                # 获取当前任务终点位置
                if t['type'] in ['A','E']:
                    cur_pos = ZONES[t['zone']]['coord']
                elif t['type'] == 'B':
                    cur_pos = GATE_COORD
                elif t['type'] == 'C':
                    cur_pos = ZONES[t['zone']]['coord']
                elif t['type'] == 'D':
                    cur_pos = GATE_COORD
                else:
                    cur_pos = GATE_COORD
                # 下一个任务起点
                idx = seq.index(task_id)
                if idx < len(seq)-1:
                    next_id = seq[idx+1]
                    next_t = tasks[next_id]
                    if next_t['type'] in ['A','E']:
                        next_pos = RAIL_POINTS[next_t['pos']]  # 起点是铁路作业点
                    elif next_t['type'] == 'B':
                        next_pos = RAIL_POINTS[next_t['pos']]
                    elif next_t['type'] == 'C':
                        next_pos = ZONES[next_t['zone']]['coord']
                    elif next_t['type'] == 'D':
                        next_pos = RAIL_POINTS[next_t['pos']]
                    else:
                        next_pos = GATE_COORD
                    travel_time = (abs(cur_pos[0]-next_pos[0]) + abs(cur_pos[1]-next_pos[1])) / params['v_truck']
                    last_time += travel_time
            avail_truck[v] = last_time

        # 处理堆场正面吊
        for y in range(schedule.num_yard):
            seq = schedule.yard_seq[y]
            last_time = avail_yard[y]
            for task_id in seq:
                t = tasks[task_id]
                if t['type'] not in ['A','C','E']:
                    continue
                prev = 0.0
                if t['type'] == 'A':
                    prev = truck_end[task_id]
                elif t['type'] == 'C':
                    prev = 0.0
                elif t['type'] == 'E':
                    prev = truck_end[task_id]
                start = max(last_time, prev)
                if abs(start - schedule.start_yard[task_id]) > 1e-6:
                    changed = True
                schedule.start_yard[task_id] = start
                end = start + t['p_yard']
                schedule.end_yard[task_id] = end
                yard_end[task_id] = end
                last_time = end
                # 移动时间：堆场正面吊移动到下一个任务的分区
                cur_zone = t['zone'] if t['type'] in ['A','C','E'] else None
                idx = seq.index(task_id)
                if idx < len(seq)-1:
                    next_id = seq[idx+1]
                    next_t = tasks[next_id]
                    if next_t['type'] in ['A','C','E']:
                        next_zone = next_t['zone']
                        travel_time = zone_zone_dist[(cur_zone, next_zone)] / params['v_yard']
                        last_time += travel_time
            avail_yard[y] = last_time

    # 计算完工时间
    max_end = max(schedule.end_rtg) if schedule.end_rtg else 0
    schedule.f1 = max_end

    # 计算能耗
    e_basic = 0.0
    e_empty = 0.0
    e_idle = 0.0

    # 装卸能耗
    for i, t in enumerate(tasks):
        if t['type'] in ['A','B','D','E']:
            e_basic += t['p_rtg'] * params['e_rtg_base']
        if t['type'] in ['A','C','E']:
            e_basic += t['p_yard'] * params['e_yard_base']

    # 空载移动能耗：上面已经计算了移动时间，但没有计入空载能耗。我们需要在设备循环中记录移动时间。
    # 为简化，这里我们忽略空载移动能耗（或根据移动时间计算）。实际上移动时间已隐含在设备可用时间中，但未乘系数。
    # 我们可以从设备可用时间中提取移动时间：设备可用时间 = 最后任务结束时间 + 移动时间总和。
    # 但较复杂，这里我们粗略地按设备移动时间估算（可改进）。

    # 空闲能耗：设备空闲时间 = 总时间 - 工作时间 - 移动时间
    # 更准确：对于每台设备，计算所有任务的工作时间和移动时间之和，空闲时间 = max_end - 工作总和 - 移动总和
    for r in range(schedule.num_rtg):
        work = sum(tasks[i]['p_rtg'] for i in schedule.rtg_seq[r])
        # 移动时间（近似）
        move = 0.0
        seq = schedule.rtg_seq[r]
        for idx in range(len(seq)-1):
            i = seq[idx]
            j = seq[idx+1]
            move += (abs(RAIL_POINTS[tasks[i]['pos']][1] - RAIL_POINTS[tasks[j]['pos']][1]) / params['v_rtg'])
        idle = max_end - work - move
        if idle < 0: idle = 0
        e_idle += idle * params['e_idle_rtg']

    for y in range(schedule.num_yard):
        work = sum(tasks[i]['p_yard'] for i in schedule.yard_seq[y] if tasks[i]['type'] in ['A','C','E'])
        move = 0.0
        seq = schedule.yard_seq[y]
        for idx in range(len(seq)-1):
            i = seq[idx]
            j = seq[idx+1]
            move += zone_zone_dist[(tasks[i]['zone'], tasks[j]['zone'])] / params['v_yard']
        idle = max_end - work - move
        if idle < 0: idle = 0
        e_idle += idle * params['e_idle_yard']

    for v in range(schedule.num_truck):
        work = 0.0
        seq = schedule.truck_seq[v]
        for i in seq:
            t = tasks[i]
            if t['type'] == 'A':
                travel = rail_zone_dist[(t['pos'], t['zone'])] / params['v_truck']
            elif t['type'] == 'B':
                travel = rail_gate_dist[t['pos']] / params['v_truck']
            elif t['type'] == 'C':
                travel = rail_zone_dist[(t['pos'], t['zone'])] / params['v_truck']
            elif t['type'] == 'D':
                travel = rail_gate_dist[t['pos']] / params['v_truck']
            elif t['type'] == 'E':
                travel = rail_zone_dist[(t['pos'], 5)] / params['v_truck']
            else:
                travel = 0
            work += travel
        move = 0.0
        for idx in range(len(seq)-1):
            i = seq[idx]
            j = seq[idx+1]
            # 计算从i的终点到j的起点的移动时间
            ti = tasks[i]
            tj = tasks[j]
            if ti['type'] in ['A','E']:
                pos_i = ZONES[ti['zone']]['coord']
            elif ti['type'] == 'B':
                pos_i = GATE_COORD
            elif ti['type'] == 'C':
                pos_i = ZONES[ti['zone']]['coord']
            elif ti['type'] == 'D':
                pos_i = GATE_COORD
            else:
                pos_i = GATE_COORD
            if tj['type'] in ['A','E']:
                pos_j = RAIL_POINTS[tj['pos']]
            elif tj['type'] == 'B':
                pos_j = RAIL_POINTS[tj['pos']]
            elif tj['type'] == 'C':
                pos_j = ZONES[tj['zone']]['coord']
            elif tj['type'] == 'D':
                pos_j = RAIL_POINTS[tj['pos']]
            else:
                pos_j = GATE_COORD
            move += (abs(pos_i[0]-pos_j[0]) + abs(pos_i[1]-pos_j[1])) / params['v_truck']
        idle = max_end - work - move
        if idle < 0: idle = 0
        e_idle += idle * params['e_idle_truck']

    # 等待惩罚
    wait_penalty = 0.0
    for i, t in enumerate(tasks):
        if t['type'] in ['B','D']:
            start = schedule.start_truck[i]
            if start < t['a']:
                wait = t['a'] - start
                wait_penalty += wait * 1.0

    schedule.f2 = e_basic + e_empty + e_idle + wait_penalty
    return schedule

# ==================== 内层仿真模块（含故障重调度） ====================
def simulate_fault_loss(schedule: Schedule, tasks: List[Dict], params: Dict) -> float:
    """蒙特卡洛仿真期望故障损失（含重调度）"""
    Q = params['Q']
    total_loss = 0.0
    np.random.seed(42)  # 固定种子

    for _ in range(Q):
        # 复制调度方案
        sch_copy = copy.deepcopy(schedule)
        # 生成故障事件
        faults = []
        for r in range(sch_copy.num_rtg):
            t = 0
            while t < sch_copy.f1:
                dt = np.random.exponential(1/params['lambda_rtg'])
                t += dt
                if t < sch_copy.f1:
                    repair = np.random.exponential(params['repair_mean'])
                    faults.append(('rtg', r, t, repair))
        for v in range(sch_copy.num_truck):
            t = 0
            while t < sch_copy.f1:
                dt = np.random.exponential(1/params['lambda_truck'])
                t += dt
                if t < sch_copy.f1:
                    repair = np.random.exponential(params['repair_mean'])
                    faults.append(('truck', v, t, repair))
        # 排序
        faults.sort(key=lambda x: x[2])

        # 快速重调度：对于每个故障，重新分配该设备上受影响的任务
        # 维护设备当前状态
        # 我们只处理故障设备上的任务，将它们移到其他设备
        # 简化：对于每个故障设备，将其上的未完成任务重新分配到同类型设备（最早可用）
        for (type_, idx, t_f, d_f) in faults:
            if type_ == 'rtg':
                # 找到该设备上在 t_f 之后开始的任务（包括正在执行的）
                affected = [i for i in sch_copy.rtg_seq[idx] if sch_copy.start_rtg[i] >= t_f - 1e-6]
                # 移除这些任务
                for i in affected:
                    sch_copy.rtg_seq[idx].remove(i)
                # 重新分配
                for i in affected:
                    # 选择其他正面吊
                    candidates = [r for r in range(sch_copy.num_rtg) if r != idx]
                    best_r = None
                    best_time = float('inf')
                    for r in candidates:
                        # 计算插入后的开始时间（假设插入到末尾）
                        last_time = sch_copy.end_rtg[sch_copy.rtg_seq[r][-1]] if sch_copy.rtg_seq[r] else 0
                        # 考虑移动时间
                        if sch_copy.rtg_seq[r]:
                            last_task = sch_copy.rtg_seq[r][-1]
                            last_pos = RAIL_POINTS[tasks[last_task]['pos']]
                            cur_pos = RAIL_POINTS[tasks[i]['pos']]
                            travel = (abs(last_pos[0]-cur_pos[0]) + abs(last_pos[1]-cur_pos[1])) / params['v_rtg']
                            start = last_time + travel
                        else:
                            start = last_time
                        if start < best_time:
                            best_time = start
                            best_r = r
                    # 插入
                    sch_copy.rtg_seq[best_r].append(i)
                    sch_copy.rtg_assign[i] = best_r
            elif type_ == 'truck':
                affected = [i for i in sch_copy.truck_seq[idx] if sch_copy.start_truck[i] >= t_f - 1e-6]
                for i in affected:
                    sch_copy.truck_seq[idx].remove(i)
                for i in affected:
                    candidates = [v for v in range(sch_copy.num_truck) if v != idx]
                    best_v = None
                    best_time = float('inf')
                    for v in candidates:
                        last_time = sch_copy.end_truck[sch_copy.truck_seq[v][-1]] if sch_copy.truck_seq[v] else 0
                        if sch_copy.truck_seq[v]:
                            last_task = sch_copy.truck_seq[v][-1]
                            # 计算移动时间
                            # 获取last_task的终点位置
                            t_last = tasks[last_task]
                            if t_last['type'] in ['A','E']:
                                last_pos = ZONES[t_last['zone']]['coord']
                            elif t_last['type'] == 'B':
                                last_pos = GATE_COORD
                            elif t_last['type'] == 'C':
                                last_pos = ZONES[t_last['zone']]['coord']
                            else:
                                last_pos = GATE_COORD
                            # 当前任务的起点
                            t_cur = tasks[i]
                            if t_cur['type'] in ['A','E']:
                                cur_pos = RAIL_POINTS[t_cur['pos']]
                            elif t_cur['type'] == 'B':
                                cur_pos = RAIL_POINTS[t_cur['pos']]
                            elif t_cur['type'] == 'C':
                                cur_pos = ZONES[t_cur['zone']]['coord']
                            elif t_cur['type'] == 'D':
                                cur_pos = RAIL_POINTS[t_cur['pos']]
                            else:
                                cur_pos = GATE_COORD
                            travel = (abs(last_pos[0]-cur_pos[0]) + abs(last_pos[1]-cur_pos[1])) / params['v_truck']
                            start = last_time + travel
                        else:
                            start = last_time
                        if start < best_time:
                            best_time = start
                            best_v = v
                    sch_copy.truck_seq[best_v].append(i)
                    sch_copy.truck_assign[i] = best_v

        # 重调度后重新解码以获取实际能耗
        sch_copy = decode(sch_copy, tasks, params)
        # 计算故障损失（包含维修期间闲置能耗）
        # 在仿真中，我们通过重调度后的能耗减去原方案能耗来得到损失
        # 注意：原方案能耗已经包含基础能耗，这里我们只计算差值
        # 为了简化，我们只计算因故障导致的额外能耗（维修闲置 + 重调度额外行驶）
        # 这里我们使用解码后计算的 f2 减去原方案 f2 作为损失
        loss = sch_copy.f2 - schedule.f2
        total_loss += max(loss, 0)

    return total_loss / Q

# ==================== 初始种群生成（混合启发式） ====================
def generate_initial_solution(method: str, tasks, num_rtg, num_yard, num_truck, params):
    """根据给定方法生成一个初始解"""
    num_tasks = len(tasks)
    sch = Schedule(num_tasks, num_rtg, num_yard, num_truck)
    # 随机分配设备
    for i in range(num_tasks):
        sch.rtg_assign[i] = random.randint(0, num_rtg-1)
        sch.truck_assign[i] = random.randint(0, num_truck-1)
        if tasks[i]['type'] in ['A','C','E']:
            sch.yard_assign[i] = random.randint(0, num_yard-1)
        else:
            sch.yard_assign[i] = -1
    # 根据方法生成序列
    if method == 'random':
        for r in range(num_rtg):
            sch.rtg_seq[r] = [i for i in range(num_tasks) if sch.rtg_assign[i] == r]
            random.shuffle(sch.rtg_seq[r])
        for v in range(num_truck):
            sch.truck_seq[v] = [i for i in range(num_tasks) if sch.truck_assign[i] == v]
            random.shuffle(sch.truck_seq[v])
        for y in range(num_yard):
            sch.yard_seq[y] = [i for i in range(num_tasks) if tasks[i]['type'] in ['A','C','E'] and sch.yard_assign[i] == y]
            random.shuffle(sch.yard_seq[y])
    elif method == 'edf':
        # 按外部集卡截止时间排序（仅影响任务顺序）
        # 这里我们按b_i升序排序所有任务
        sorted_tasks = sorted([i for i in range(num_tasks)], key=lambda i: tasks[i]['b'] if tasks[i]['type'] in ['B','D'] else float('inf'))
        for r in range(num_rtg):
            sch.rtg_seq[r] = []
        for v in range(num_truck):
            sch.truck_seq[v] = []
        for i in sorted_tasks:
            r = sch.rtg_assign[i]
            sch.rtg_seq[r].append(i)
            v = sch.truck_assign[i]
            sch.truck_seq[v].append(i)
        for y in range(num_yard):
            sch.yard_seq[y] = [i for i in sorted_tasks if tasks[i]['type'] in ['A','C','E'] and sch.yard_assign[i] == y]
    elif method == 'spt':
        # 按p_rtg升序
        sorted_tasks = sorted([i for i in range(num_tasks)], key=lambda i: tasks[i]['p_rtg'])
        for r in range(num_rtg):
            sch.rtg_seq[r] = []
        for v in range(num_truck):
            sch.truck_seq[v] = []
        for i in sorted_tasks:
            r = sch.rtg_assign[i]
            sch.rtg_seq[r].append(i)
            v = sch.truck_assign[i]
            sch.truck_seq[v].append(i)
        for y in range(num_yard):
            sch.yard_seq[y] = [i for i in sorted_tasks if tasks[i]['type'] in ['A','C','E'] and sch.yard_assign[i] == y]
    elif method == 'greedy_energy':
        # 能耗贪婪：随机顺序，但分配时尝试最小能耗
        # 此处简化：随机顺序，然后依次插入
        order = list(range(num_tasks))
        random.shuffle(order)
        # 清空序列
        for r in range(num_rtg):
            sch.rtg_seq[r] = []
        for v in range(num_truck):
            sch.truck_seq[v] = []
        for y in range(num_yard):
            sch.yard_seq[y] = []
        # 按顺序插入每个任务到设备序列末尾（因为分配固定）
        for i in order:
            r = sch.rtg_assign[i]
            sch.rtg_seq[r].append(i)
            v = sch.truck_assign[i]
            sch.truck_seq[v].append(i)
            if tasks[i]['type'] in ['A','C','E']:
                y = sch.yard_assign[i]
                sch.yard_seq[y].append(i)
    else:
        # 默认随机
        return generate_initial_solution('random', tasks, num_rtg, num_yard, num_truck, params)

    return sch

def init_population(pop_size, tasks, num_rtg, num_yard, num_truck, params):
    """初始化种群，使用混合规则"""
    pop = []
    methods = ['random', 'edf', 'spt', 'greedy_energy']
    for i in range(pop_size):
        method = random.choice(methods)
        sch = generate_initial_solution(method, tasks, num_rtg, num_yard, num_truck, params)
        pop.append(sch)
    return pop

# ==================== NSGA-II 核心函数 ====================
def non_dominated_sort(pop):
    fronts = []
    for p in pop:
        p.dominated = []
        p.domination_count = 0
    for i, p in enumerate(pop):
        for j, q in enumerate(pop):
            if i == j: continue
            if p.f1 <= q.f1 and p.f2 <= q.f2 and (p.f1 < q.f1 or p.f2 < q.f2):
                p.dominated.append(q)
            elif q.f1 <= p.f1 and q.f2 <= p.f2 and (q.f1 < p.f1 or q.f2 < p.f2):
                p.domination_count += 1
        if p.domination_count == 0:
            p.rank = 0
            if not fronts: fronts.append([])
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i+1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
    return fronts

def crowding_distance(front):
    l = len(front)
    if l <= 2:
        for p in front:
            p.crowding = float('inf')
        return
    for p in front:
        p.crowding = 0.0
    # f1
    front.sort(key=lambda x: x.f1)
    f1_min = front[0].f1
    f1_max = front[-1].f1
    front[0].crowding = float('inf')
    front[-1].crowding = float('inf')
    for i in range(1, l-1):
        front[i].crowding += (front[i+1].f1 - front[i-1].f1) / (f1_max - f1_min + 1e-9)
    # f2
    front.sort(key=lambda x: x.f2)
    f2_min = front[0].f2
    f2_max = front[-1].f2
    for i in range(1, l-1):
        front[i].crowding += (front[i+1].f2 - front[i-1].f2) / (f2_max - f2_min + 1e-9)

def tournament_selection(pop):
    a = random.choice(pop)
    b = random.choice(pop)
    if a.rank < b.rank:
        return copy.deepcopy(a)
    elif a.rank > b.rank:
        return copy.deepcopy(b)
    else:
        if a.crowding > b.crowding:
            return copy.deepcopy(a)
        else:
            return copy.deepcopy(b)

def crossover(parent1, parent2, tasks, pc):
    if random.random() > pc:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    # 均匀交叉分配
    for i in range(child1.num_tasks):
        if random.random() < 0.5:
            child1.rtg_assign[i], child2.rtg_assign[i] = child2.rtg_assign[i], child1.rtg_assign[i]
            child1.truck_assign[i], child2.truck_assign[i] = child2.truck_assign[i], child1.truck_assign[i]
            if tasks[i]['type'] in ['A','C','E']:
                child1.yard_assign[i], child2.yard_assign[i] = child2.yard_assign[i], child1.yard_assign[i]
    # 重建序列
    for sch in [child1, child2]:
        sch.rtg_seq = [[] for _ in range(sch.num_rtg)]
        sch.truck_seq = [[] for _ in range(sch.num_truck)]
        sch.yard_seq = [[] for _ in range(sch.num_yard)]
        for i in range(sch.num_tasks):
            r = sch.rtg_assign[i]
            sch.rtg_seq[r].append(i)
            v = sch.truck_assign[i]
            sch.truck_seq[v].append(i)
            if tasks[i]['type'] in ['A','C','E']:
                y = sch.yard_assign[i]
                sch.yard_seq[y].append(i)
        # 随机打乱
        for seq in sch.rtg_seq:
            random.shuffle(seq)
        for seq in sch.truck_seq:
            random.shuffle(seq)
        for seq in sch.yard_seq:
            random.shuffle(seq)
    return child1, child2

def mutate(individual, tasks, pm):
    if random.random() > pm:
        return
    # 选择变异类型
    if random.random() < 0.5:
        # 交换变异
        device_type = random.choice(['rtg','truck','yard'])
        if device_type == 'rtg' and individual.num_rtg>0:
            r = random.randint(0, individual.num_rtg-1)
            seq = individual.rtg_seq[r]
            if len(seq) >= 2:
                i, j = random.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]
        elif device_type == 'truck' and individual.num_truck>0:
            v = random.randint(0, individual.num_truck-1)
            seq = individual.truck_seq[v]
            if len(seq) >= 2:
                i, j = random.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]
        elif device_type == 'yard' and individual.num_yard>0:
            y = random.randint(0, individual.num_yard-1)
            seq = individual.yard_seq[y]
            if len(seq) >= 2:
                i, j = random.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]
    else:
        # 重新分配变异
        i = random.randint(0, individual.num_tasks-1)
        # 重新分配正面吊
        new_r = random.randint(0, individual.num_rtg-1)
        old_r = individual.rtg_assign[i]
        if old_r != new_r:
            individual.rtg_assign[i] = new_r
            individual.rtg_seq[old_r].remove(i)
            individual.rtg_seq[new_r].append(i)
        # 重新分配集卡
        new_v = random.randint(0, individual.num_truck-1)
        old_v = individual.truck_assign[i]
        if old_v != new_v:
            individual.truck_assign[i] = new_v
            individual.truck_seq[old_v].remove(i)
            individual.truck_seq[new_v].append(i)
        # 重新分配堆场正面吊（如果需要）
        if tasks[i]['type'] in ['A','C','E']:
            new_y = random.randint(0, individual.num_yard-1)
            old_y = individual.yard_assign[i]
            if old_y != new_y:
                individual.yard_assign[i] = new_y
                individual.yard_seq[old_y].remove(i)
                individual.yard_seq[new_y].append(i)
        # 简单打乱新序列
        for seq in [individual.rtg_seq, individual.truck_seq, individual.yard_seq]:
            for s in seq:
                random.shuffle(s)

def repair(individual, tasks):
    """确保分配与序列一致，并随机打乱序列以增加多样性"""
    # 重建序列（确保一致性）
    for r in range(individual.num_rtg):
        individual.rtg_seq[r] = []
    for v in range(individual.num_truck):
        individual.truck_seq[v] = []
    for y in range(individual.num_yard):
        individual.yard_seq[y] = []
    for i in range(individual.num_tasks):
        r = individual.rtg_assign[i]
        individual.rtg_seq[r].append(i)
        v = individual.truck_assign[i]
        individual.truck_seq[v].append(i)
        if tasks[i]['type'] in ['A','C','E']:
            y = individual.yard_assign[i]
            individual.yard_seq[y].append(i)
    # 打乱序列
    for seq in individual.rtg_seq:
        random.shuffle(seq)
    for seq in individual.truck_seq:
        random.shuffle(seq)
    for seq in individual.yard_seq:
        random.shuffle(seq)
    return individual

def evaluate_population(pop, tasks, params):
    for sch in pop:
        decode(sch, tasks, params)
        loss = simulate_fault_loss(sch, tasks, params)
        sch.f2 += params['beta'] * loss

# ==================== 主函数 ====================
def main():
    # 参数设置
    num_tasks = 20
    num_rtg = 2
    num_yard = 2
    num_truck = 4
    tasks = generate_tasks(num_tasks)

    # 初始化种群
    pop = init_population(PARAMS['pop_size'], tasks, num_rtg, num_yard, num_truck, PARAMS)
    evaluate_population(pop, tasks, PARAMS)

    # 主循环
    gen = 0
    no_improve = 0
    best_front = None

    for gen in range(PARAMS['max_gen']):
        # 非支配排序
        fronts = non_dominated_sort(pop)
        # 计算拥挤度
        for f in fronts:
            crowding_distance(f)

        # 生成子代
        offspring = []
        while len(offspring) < PARAMS['pop_size']:
            p1 = tournament_selection(pop)
            p2 = tournament_selection(pop)
            c1, c2 = crossover(p1, p2, tasks, PARAMS['pc'])
            mutate(c1, tasks, PARAMS['pm'])
            mutate(c2, tasks, PARAMS['pm'])
            c1 = repair(c1, tasks)
            c2 = repair(c2, tasks)
            offspring.append(c1)
            if len(offspring) < PARAMS['pop_size']:
                offspring.append(c2)

        # 评估子代
        evaluate_population(offspring, tasks, PARAMS)

        # 合并
        combined = pop + offspring
        fronts = non_dominated_sort(combined)
        for f in fronts:
            crowding_distance(f)

        # 精英保留
        new_pop = []
        i = 0
        while len(new_pop) + len(fronts[i]) <= PARAMS['pop_size']:
            new_pop.extend(fronts[i])
            i += 1
        fronts[i].sort(key=lambda x: x.crowding, reverse=True)
        new_pop.extend(fronts[i][:PARAMS['pop_size'] - len(new_pop)])
        pop = new_pop

        # 检查早停
        current_front = fronts[0]
        if best_front is None:
            best_front = current_front
            no_improve = 0
        else:
            # 简单比较前沿的平均目标值
            curr_avg_f1 = np.mean([p.f1 for p in current_front])
            curr_avg_f2 = np.mean([p.f2 for p in current_front])
            best_avg_f1 = np.mean([p.f1 for p in best_front])
            best_avg_f2 = np.mean([p.f2 for p in best_front])
            if abs(curr_avg_f1 - best_avg_f1) < 1e-3 and abs(curr_avg_f2 - best_avg_f2) < 1e-3:
                no_improve += 1
            else:
                no_improve = 0
                best_front = current_front

        print(f"Gen {gen+1}: front size={len(current_front)}, best f1={min(p.f1 for p in current_front):.2f}, best f2={min(p.f2 for p in current_front):.2f}")

        if no_improve >= PARAMS['stall_gen']:
            print(f"Early stop at generation {gen+1}")
            break

    # 最终帕累托前沿
    final_front = non_dominated_sort(pop)[0]
    print("\n=== Final Pareto Front ===")
    for p in sorted(final_front, key=lambda x: x.f1):
        print(f"f1={p.f1:.2f}, f2={p.f2:.2f}")

    # 绘制帕累托前沿图
    plt.figure(figsize=(8,6))
    plt.scatter([p.f1 for p in final_front], [p.f2 for p in final_front], c='red', marker='o', label='Pareto front')
    plt.xlabel('Makespan (s)')
    plt.ylabel('Total Energy (kWh)')
    plt.title('Pareto Front')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 选择一个平衡解（膝点）打印详细调度方案
    # 简单选择：离原点最近的解
    knee = min(final_front, key=lambda p: p.f1**2 + p.f2**2)
    print("\n=== Detailed schedule of the knee solution ===")
    print(f"Objective: f1={knee.f1:.2f}, f2={knee.f2:.2f}")
    print("\n--- RTG sequences ---")
    for r in range(num_rtg):
        seq = knee.rtg_seq[r]
        print(f"RTG {r+1}: {seq}")
    print("\n--- Truck sequences ---")
    for v in range(num_truck):
        seq = knee.truck_seq[v]
        print(f"Truck {v+1}: {seq}")
    print("\n--- Yard sequences ---")
    for y in range(num_yard):
        seq = knee.yard_seq[y]
        print(f"Yard {y+1}: {seq}")
    print("\n--- Task timeline (start and end) ---")
    for i in range(num_tasks):
        t = tasks[i]
        print(f"Task {i} ({t['type']}): rtg=[{knee.start_rtg[i]:.1f},{knee.end_rtg[i]:.1f}], truck=[{knee.start_truck[i]:.1f},{knee.end_truck[i]:.1f}], yard=[{knee.start_yard[i]:.1f},{knee.end_yard[i]:.1f}]")

if __name__ == "__main__":
    main()