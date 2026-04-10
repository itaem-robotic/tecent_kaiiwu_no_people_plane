#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery feature preprocessor.
智运无人机特征预处理器。
"""


import numpy as np
from agent_ppo.conf.conf import Config


def norm(v, max_v, min_v=0):
    """Normalize v to [0, 1].

    将 v 归一化到 [0, 1]。
    """
    v = np.clip(v, min_v, max_v)
    return (v - min_v) / (max_v - min_v)


#描述驿站的7个特征：是否存在、相对位置（x,y）、绝对位置（x,y）、距离、是否为目标
def _get_pos_feature(found, cur_pos, target_pos, is_target=False):
    """Compute 7D position feature for a target relative to current position.

    计算目标位置相对于当前位置的 7 维特征。
    """
    relative_pos = (target_pos[0] - cur_pos[0], target_pos[1] - cur_pos[1])
    dist = np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
    abs_norm = norm(np.array(target_pos), 128, 0)
    return np.array(
        [
            float(found),
            norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
            norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
            abs_norm[0],
            abs_norm[1],
            norm(dist, 1.41 * 128),
            1.0 if is_target else 0.0,
        ]
    )

import heapq
from typing import List, Tuple, Optional, Set

class AStarPlanner:
    # 动作映射：dx, dz, action_index (注意 z 向下为正)
    MOVES = [
        ( 1,  0, 0),  # 右
        ( 1, -1, 1),  # 右上
        ( 0, -1, 2),  # 上
        (-1, -1, 3),  # 左上
        (-1,  0, 4),  # 左
        (-1,  1, 5),  # 左下
        ( 0,  1, 6),  # 下
        ( 1,  1, 7),  # 右下
    ]

    def __init__(self, grid_size: int = 128):
        self.grid_size = grid_size          # 地图尺寸 128x128
        self.width = grid_size
        self.height = grid_size
        self.obstacles: Set[Tuple[int, int]] = set()
        self.known_grids: Set[Tuple[int, int]] = set()

    def is_valid(self, gx: int, gz: int) -> bool:
        """检查网格是否在地图内且可通行"""
        if not (0 <= gx < self.width and 0 <= gz < self.height):
            return False
        return (gx, gz) not in self.obstacles

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = abs(a[0] - b[0])
        dz = abs(a[1] - b[1])
        return (dx + dz) + (1.414 - 2) * min(dx, dz)

    # def update_from_local_map(self, center_gx, center_gz, local_map):
    #     half = len(local_map) // 2
    #     for dz in range(-half, half + 1):
    #         for dx in range(-half, half + 1):
    #             gx = center_gx + dx
    #             gz = center_gz + dz
    #             if 0 <= gx < self.width and 0 <= gz < self.height:
    #                 self.known_grids.add((gx, gz))       # 标记已知
    #                 if local_map[dz+half][dx+half] == 0:
    #                     self.obstacles.add((gx, gz))
    #                 else:
    #                     self.obstacles.discard((gx, gz))

    def plan(self, start_grid: Tuple[int, int], goal_grid: Tuple[int, int]) -> Optional[List[int]]:
        """返回从起点到终点的动作索引列表"""
        start = start_grid
        goal = goal_grid
        if not self.is_valid(*start) or not self.is_valid(*goal):
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from: dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
        g_score: dict[Tuple[int, int], float] = {start: 0.0}
        f_score: dict[Tuple[int, int], float] = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                actions = []
                while current in came_from:
                    prev, act = came_from[current]
                    actions.append(act)
                    current = prev
                actions.reverse()
                return actions
            for dx, dz, act in self.MOVES:
                neighbor = (current[0] + dx, current[1] + dz)
                if not self.can_move(current, act):
                    continue
                tentative_g = g_score[current] + (1.414 if dx != 0 and dz != 0 else 1.0)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, act)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def next_action(self, start_grid: Tuple[int, int], goal_grid: Tuple[int, int]) -> Optional[int]:
        actions = self.plan(start_grid, goal_grid)
        if actions and len(actions) > 0:
            return actions[0]
        return None

    def is_action_safe(self, start_grid: Tuple[int, int], action: int, local_map: List[List[int]], half=10) -> bool:
        if action < 0 or action >= len(self.MOVES):
            return False
        dx, dz, _ = self.MOVES[action]
        next_gx = start_grid[0] + dx
        next_gz = start_grid[1] + dz
        row = dz + half
        col = dx + half
         # 检查目标格子是否在局部地图内且可通行
        if not (0 <= row < len(local_map) and 0 <= col < len(local_map[0])):
            return False
        if local_map[row][col] == 0:
            return False
        # 斜向移动防穿角检查（基于局部地图）
        if dx != 0 and dz != 0:
            # 检查两个相邻直线格子的局部地图值（注意坐标转换）
            # 相邻1: (dx, 0)
            row1 = 0 + half
            col1 = dx + half
            # 相邻2: (0, dz)
            row2 = dz + half
            col2 = 0 + half
            # 边界检查
            if not (0 <= row1 < len(local_map) and 0 <= col1 < len(local_map[0])):
                return False
            if not (0 <= row2 < len(local_map) and 0 <= col2 < len(local_map[0])):
                return False
            if local_map[row1][col1] == 0 and local_map[row2][col2] == 0:
                return False
        return True
    #将官方无人机视为移动障碍物，更新 obstacles 集合
    def update_from_local_map(self, center_gx, center_gz, local_map, npc_grids=None):
        half = len(local_map) // 2
        # 先清除上一次的 NPC 障碍物（可选，避免累积）
        # 但更简单：每次重新构建 obstacles，基于静态地图 + 当前 NPC
        # 注意：不能直接清空 obstacles，因为静态障碍物需要保留
        # 建议：先保存静态障碍物（从 local_map 读），再添加 npc_grids
        static_obstacles = set()
        for dz in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gx = center_gx + dx
                gz = center_gz + dz
                if 0 <= gx < self.width and 0 <= gz < self.height:
                    self.known_grids.add((gx, gz))
                    if local_map[dz+half][dx+half] == 0:
                        static_obstacles.add((gx, gz))
        # 更新 obstacles = 静态障碍物 + 当前 NPC 网格
        self.obstacles = static_obstacles.copy()
        if npc_grids:
            for ng in npc_grids:
                if self.is_valid(*ng):   # 只添加有效网格
                    self.obstacles.add(ng)
    
    def can_move(self, from_grid: Tuple[int, int], action: int) -> bool:
        """检查从当前网格执行动作是否合法（含防穿角）。"""
        dx, dz, _ = self.MOVES[action]
        to_grid = (from_grid[0] + dx, from_grid[1] + dz)
        # 目标格子必须可通行
        if not self.is_valid(*to_grid):
            return False
        # 斜向移动需额外检查防穿角
        if dx != 0 and dz != 0:
            # 两个相邻直线格子
            adj1 = (from_grid[0] + dx, from_grid[1])
            adj2 = (from_grid[0], from_grid[1] + dz)
            if not (self.is_valid(*adj1) or self.is_valid(*adj2)):
                return False
        return True
#预处理器类，包含状态重置、观察解析、特征提取、合法动作获取和奖励计算等方法
class Preprocessor:
    """feature preprocessor for Drone Delivery.

    智运无人机预处理器，仅保留最少信息。
    """

    def __init__(self, use_astar_feature=True, use_astar_reward=True):
        
        # 初始化 A* 规划器（地图边界根据环境设定，这里用 [-128, 128]）
        self.astar = AStarPlanner(grid_size=128)
        self.use_astar_feature = use_astar_feature
        self.use_astar_reward = use_astar_reward
        self.episode_count = 0
        self.astar_reward_coef = 1   # 初始跟随奖励系数
        self.Punishment_coef = 1   # 惩罚系数
        self.reset()
    def reset(self):
        """Reset all internal state.

        重置所有状态。
        """
        self.cur_pos = (0, 0)

        # Game state / 游戏状态
        self.battery = 100
        self.battery_max = 100
        self.packages = []
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0

        # Entities / 实体
        self.stations = []

        self.last_pos = None           # 上一步位置
        self.last_dist_to_target = None # 上一步到目标驿站的距离
        self.stuck_steps = 0           # 连续停滞步数
        
        self.charging_stations = []      # 充电桩列表
        self.npc_drones = []             # 官方无人机列表
        self.charge_loiter_steps = 0     # 充电桩附近停留步数计数      
        self.warehouse_pos = None   # 仓库坐标 (x, z) 
        self.warehouse_loiter_steps = 0   # 仓库附近停留步数
        self.last_in_warehouse = False    # 上一步是否在仓库区域    
        self.last_in_warehouse_for_reward = False   # 记录上一步是否在仓库内（用于进入奖励）
        self.episode_count += 1
        if self.episode_count % 2000 == 0:
            self.astar_reward_coef = max(0.5, self.astar_reward_coef * 0.98)
            self.Punishment_coef = max(0.5, self.Punishment_coef * 0.98)

        self.last_package_count = 0
        self.last_min_npc_dist = None

    def _world_to_grid(self, wx: float, wz: float) -> Tuple[int, int]:
        """世界坐标 → 网格索引（直接取整，因为网格边长为1）"""
        return (int(round(wx)), int(round(wz)))

    def _grid_to_world(self, gx: int, gz: int) -> Tuple[float, float]:
        """网格索引 → 世界坐标（网格中心）"""
        return (float(gx), float(gz))

    def _parse_obs(self, env_obs):
        """Parse essential fields from observation dict.

        从 observation 字典中解析必要字段。
        """
        obs = env_obs["observation"]
        frame_state = obs["frame_state"]
        # 保存当前局部地图（21x21），用于后续 A* 安全性检查
        
        hero = frame_state["heroes"]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        # self.last_package_count = getattr(self, 'last_package_count', 0)
        # self.package_count = len(self.packages)
        self.last_package_count = len(self.packages)   # 注意：这里使用更新前的 self.package

        self.battery = hero.get("battery", self.battery_max)
        self.battery_max = hero.get("battery_max", 100)
        self.packages = hero.get("packages", [])

        self.package_count = len(self.packages)

        self.last_delivered = self.delivered
        self.delivered = hero.get("delivered", 0)
        self.step_no = obs.get("step_no", 0)

        self.stations = []
        self.charging_stations = []
        for organ in frame_state.get("organs", []):
            sub_type = organ.get("sub_type", 0)
            if sub_type == 3:
                self.stations.append(organ)
            elif sub_type == 2:
                self.charging_stations.append(organ)
            elif sub_type == 1:    # 仓库 sub_type=1
                self.warehouse_pos = (organ["pos"]["x"], organ["pos"]["z"])

        if self.warehouse_pos is None:
            # 默认使用地图中心 (64,64)
            self.warehouse_pos = (64.0, 64.0)

        self.npc_drones = []
        for npc in frame_state.get("npcs", []):
            # 根据实际字段判断，此处假设所有 npcs 均为无人机
            self.npc_drones.append(npc)

        self.legal_act = obs.get("legal_action", [1] * 8)
        map_info = obs.get("map_info")

        self.npc_grids = []
        for npc in self.npc_drones:
            nx, nz = self._world_to_grid(npc["pos"]["x"], npc["pos"]["z"])
            self.npc_grids.append((nx, nz))

        if map_info is not None:
            gx, gz = self._world_to_grid(*self.cur_pos)
            # 将 NPC 网格膨胀为 5×5 区域（视为更大障碍物）
            expanded_npc_grids = []
            for (nx, nz) in self.npc_grids:
                for dx in range(-2, 3):      # -2, -1, 0, 1, 2
                    for dz in range(-2, 3):
                        expanded_npc_grids.append((nx + dx, nz + dz))
            self.astar.update_from_local_map(gx, gz, map_info, expanded_npc_grids)
            
            #self.astar.update_from_local_map(gx, gz, map_info)
        self.map_info = map_info if map_info is not None else np.ones((21, 21), dtype=int)
        # 收集 NPC 无人机网格坐标（世界坐标转网格）


    def feature_process(self, env_obs, last_action):
        """Core feature extraction. Returns (feature_22d, legal_action, reward).

        核心特征提取方法，返回 22 维特征向量、合法动作掩码和奖励。
        """
        self._parse_obs(env_obs)
        cur_pos = self.cur_pos   # 已经更新为最新位置
        # 1. Hero state features (4D) / 英雄状态特征（4D）
        battery_ratio = norm(self.battery, self.battery_max)
        package_count_norm = norm(len(self.packages), 3)
        cur_pos_norm = norm(np.array(self.cur_pos, dtype=float), 128, 0)
        hero_feat = np.array(
            [
                battery_ratio,
                package_count_norm,
                cur_pos_norm[0],
                cur_pos_norm[1],
            ]
        )

        # 2. Nearest 1 station feature (7D) / 最近 1 个驿站特征（7D）
        # Target stations first, then by distance
        # 目标驿站优先，然后按距离排序
        # 2. Nearest 1 station feature (7D) and A* guided action
        target_ids = set(self.packages)
        target_pos = self._get_current_target_position()

        # A* 推荐动作（默认 None）
        astar_recommended_action = None

        if target_pos is not None:
            start_grid = self._world_to_grid(*cur_pos)
            global_goal_grid = self._world_to_grid(*target_pos)
            dist_to_goal = max(abs(global_goal_grid[0] - start_grid[0]), abs(global_goal_grid[1] - start_grid[1]))
        #    先尝试直接规划到全局目标
            #action = self.astar.next_action(start_grid, global_goal_grid)
            LOCAL_THRESHOLD = 10
            use_local_subgoal = (dist_to_goal > LOCAL_THRESHOLD)
            action = None
            if use_local_subgoal:
                # 优先尝试局部子目标
                local_goal = self._get_local_subgoal(start_grid, global_goal_grid)
                if local_goal is not None:
                    action = self.astar.next_action(start_grid, local_goal)
            # 如果规划失败，且全局目标在视野外（超出10格），尝试局部子目标
            # if action is None:
            #     max_diff = max(abs(global_goal_grid[0] - start_grid[0]),
            #                 abs(global_goal_grid[1] - start_grid[1]))
            #     if max_diff > 10:
            #         local_goal = self._get_local_subgoal(start_grid, global_goal_grid)
            #         if local_goal is not None:
            #             action = self.astar.next_action(start_grid, local_goal)
            if action is None:
                action = self.astar.next_action(start_grid, global_goal_grid)
                # 如果全局规划也失败，且目标距离 > 10，可以再次尝试局部子目标（fallback）
                if action is None and dist_to_goal > LOCAL_THRESHOLD:
                    local_goal = self._get_local_subgoal(start_grid, global_goal_grid)
                    if local_goal is not None:
                        action = self.astar.next_action(start_grid, local_goal)
            # ----- 安全性验证（使用当前局部地图）-----
            if action is not None:
                # 检查该动作下一步是否在局部地图内且可通行
                if not self.astar.is_action_safe(start_grid, action, self.map_info, half=10):
                    # 不安全：丢弃推荐动作
                    action = None
            if action is None:
            # 尝试紧急逃离动作
                action = self.get_escape_action()
                # 如果逃离动作也无效（例如无无人机），则随机选择一个合法动作
                if action is None:
                    legal = self._get_legal_action()
                    # 从合法动作中随机选一个
                    action = np.random.choice([i for i, v in enumerate(legal) if v == 1])
            
        # 可选：再验证动作是否合法（如果 legal_action 已经算出，可提前过滤）
        # 但 legal_action 还未在此处计算，可以稍后做，或者信任环境提供的掩码
            astar_recommended_action = action

        # 构建 A* 特征（8维 one-hot）
        astar_action_feat = np.zeros(8, dtype=float)
        if astar_recommended_action is not None:
            astar_action_feat[astar_recommended_action] = 1.0

        def station_sort_key(s):
            is_tgt = s.get("config_id", 0) in target_ids
            dist = np.sqrt((s["pos"]["x"] - self.cur_pos[0]) ** 2 + (s["pos"]["z"] - self.cur_pos[1]) ** 2)
            return (0 if is_tgt else 1, dist)

        sorted_stations = sorted(self.stations, key=station_sort_key)

        if len(sorted_stations) > 0:
            s = sorted_stations[0]
            is_target = s.get("config_id", 0) in target_ids
            station_feat = _get_pos_feature(
                True,
                self.cur_pos,
                (s["pos"]["x"], s["pos"]["z"]),
                is_target=is_target,
            )
            target_visible = float(is_target)
        else:
            station_feat = _get_pos_feature(False, self.cur_pos, self.cur_pos, is_target=False)
            target_visible = 0.0

        # 3. Legal action mask (8D) / 合法动作掩码（8D）
        legal_action = self._get_legal_action()

        # 4. Binary indicators (3D) / 二值指示器（3D）
        has_package = 1.0 if len(self.packages) > 0 else 0.0
        battery_low = 1.0 if (self.battery / max(self.battery_max, 1)) < 0.3 else 0.0
        indicators = np.array([has_package, battery_low, target_visible])


         # 5. 官方无人机特征（3D）
        npc_feat = np.zeros(3, dtype=float)
        if self.npc_drones:
            # 找最近的无人机
            min_dist = float('inf')
            closest_npc_pos = None
            for npc in self.npc_drones:
                npc_pos = (npc["pos"]["x"], npc["pos"]["z"])
                dist = np.hypot(self.cur_pos[0] - npc_pos[0], self.cur_pos[1] - npc_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_npc_pos = npc_pos
         # 特征0：是否存在（1.0）
            npc_feat[0] = 1.0
            # 特征1：归一化距离（0~1，距离为0时1，距离>=10时0）
            npc_feat[1] = 1.0 - min(min_dist / 10.0, 1.0)
            # 特征2：相对方向点积（归一化到[-1,1]后再映射到[0,1]）
            if closest_npc_pos is not None:
                dx = closest_npc_pos[0] - self.cur_pos[0]
                dz = closest_npc_pos[1] - self.cur_pos[1]
                if min_dist > 1e-4:
                    dir_x = dx / min_dist
                    dir_z = dz / min_dist
                    # 点积：与当前移动方向？没有移动方向，可以用固定参考（例如朝右）
                    # 更简单：直接使用 (dir_x, dir_z) 作为两个特征，但为了维度固定，我们使用点积与一个固定方向（如朝向目标）
                    # 或者将 (dir_x, dir_z) 直接作为两个特征，但那样会增加维度。此处简化为：与朝向目标的方向点积
                    target_pos = self._get_current_target_position()
                    if target_pos is not None:
                        tdx = target_pos[0] - self.cur_pos[0]
                        tdz = target_pos[1] - self.cur_pos[1]
                        tdist = np.hypot(tdx, tdz)
                        if tdist > 1e-4:
                            tdir_x = tdx / tdist
                            tdir_z = tdz / tdist
                            dot = dir_x * tdir_x + dir_z * tdir_z   # -1..1
                            npc_feat[2] = (dot + 1.0) / 2.0        # 映射到 0..1
                        else:
                            npc_feat[2] = 0.5
                    else:
                        npc_feat[2] = 0.5
        else:
            npc_feat[0] = 0.0
            npc_feat[1] = 0.0
            npc_feat[2] = 0.5


        if self.use_astar_feature:
            feature = np.concatenate([
                hero_feat,
                station_feat,
                np.array(legal_action, dtype=float),
                indicators,
                astar_action_feat,
                npc_feat              # 新增 3 维 → 总维度变为 33
            ])
        else:
            feature = np.concatenate([
                hero_feat,
                station_feat,
                np.array(legal_action, dtype=float),
                indicators,
                npc_feat              # 新增 3 维 → 总维度变为 22 + 3 = 25
            ])

       

        # 计算奖励（传入 last_action 和 astar 推荐动作）
        stuck_flag = self.stuck_steps >= 5
        reward = self._reward_process(last_action, legal_action, astar_recommended_action)
        return feature, legal_action, reward, stuck_flag




    def _get_min_npc_distance(self):
        """返回最近官方无人机的距离（若无则返回大数）"""
        if not self.npc_drones:
            return 999.0
        min_dist = min(
            np.hypot(self.cur_pos[0] - npc["pos"]["x"], self.cur_pos[1] - npc["pos"]["z"])
            for npc in self.npc_drones
        )
        return min_dist
    def get_escape_action(self):
        if not self.npc_drones:
            return None
        # 找最近的无人机
        min_dist = float('inf')
        closest_dir = (0, 0)
        for npc in self.npc_drones:
            npc_pos = (npc["pos"]["x"], npc["pos"]["z"])
            dx = self.cur_pos[0] - npc_pos[0]
            dz = self.cur_pos[1] - npc_pos[1]
            dist = np.hypot(dx, dz)
            if dist < min_dist:
                min_dist = dist
                if dist > 1e-4:
                    closest_dir = (dx / dist, dz / dist)
        # 动作方向向量（归一化）
        action_dirs = [
            (1,0), (0.707,-0.707), (0,-1), (-0.707,-0.707),
            (-1,0), (-0.707,0.707), (0,1), (0.707,0.707)
        ]
        # 选择与远离方向最接近的动作
        best_action = 0
        best_dot = -2
        for i, (ax, az) in enumerate(action_dirs):
            dot = ax * closest_dir[0] + az * closest_dir[1]
            if dot > best_dot:
                best_dot = dot
                best_action = i
        return best_action

    def _get_legal_action(self):
        """Get legal action mask.

        获取合法动作掩码。
        """
        if hasattr(self, "legal_act") and self.legal_act:
            legal_action = [int(x) for x in self.legal_act[:8]]
        else:
            legal_action = [1] * 8

        if sum(legal_action) == 0:
            return [1] * 8

        return legal_action

    def _get_local_subgoal(self, cur_grid, global_goal_grid, half=10):
        """
        在局部视野边界寻找朝向全局目标的、已知可通行的子目标网格。
        返回 (subgoal_grid) 或 None。
        """
        cx, cz = cur_grid
        gx, gz = global_goal_grid

    # 方向向量（未归一化）
        dx_global = gx - cx
        dz_global = gz - cz

    # 收集边界候选
        candidates = []
        for nx in range(cx - half, cx + half + 1):
            for nz in range(cz - half, cz + half + 1):
                # 只考虑边界（矩形边界）
                if max(abs(nx - cx), abs(nz - cz)) != half:
                    continue
                # 必须在地图内、已知且可通行
                if not self.astar.is_valid(nx, nz):
                    continue
                if (nx, nz) not in self.astar.known_grids:
                    continue
                obstacle_count = 0
                for ddx in [-1, 0, 1]:
                    for ddz in [-1, 0, 1]:
                        if ddx == 0 and ddz == 0:
                            continue
                        if not self.astar.is_valid(nx + ddx, nz + ddz):
                            obstacle_count += 1
                if obstacle_count >= 3:
                    continue

                # 计算与全局目标的距离（用于排序）
                dist_to_goal = (nx - gx)**2 + (nz - gz)**2
                # 计算与期望方向的夹角（可选，作为辅助排序）
                dot = (nx - cx) * dx_global + (nz - cz) * dz_global
                candidates.append((dist_to_goal, -dot, (nx, nz)))  # 距离小优先，dot大优先

        if not candidates:
            return None

    # 排序：优先距离目标近的，其次方向一致
        candidates.sort()
        return candidates[0][2]

    def _get_current_target_position(self) -> Optional[Tuple[float, float]]:
        """返回当前需要前往的目标驿站坐标（世界坐标）"""
        LOW_BATTERY_THRESHOLD = 0.4
        battery_ratio = self.battery / max(self.battery_max, 1)
        if battery_ratio < LOW_BATTERY_THRESHOLD:
            charge_points = []
            if self.warehouse_pos is not None:
                charge_points.append(self.warehouse_pos)
            for cs in self.charging_stations:
                charge_points.append((cs["pos"]["x"], cs["pos"]["z"]))
            if charge_points:
                nearest = min(charge_points,
                            key=lambda p: np.hypot(self.cur_pos[0]-p[0], self.cur_pos[1]-p[1]))
                return (nearest[0], nearest[1])

        # 如果有携带包裹，目标为第一个包裹对应的驿站
        # 与奖励函数一致
        if self.packages:
            target_id = self.packages[0]
            for s in self.stations:
                if s.get("config_id") == target_id:
                    return (s["pos"]["x"], s["pos"]["z"])
            return None

        # 3. 无包裹 + 电量充足：去仓库取货
        if self.warehouse_pos is not None:
            return (self.warehouse_pos[0], self.warehouse_pos[1])
        return (64.0, 64.0)


    
    def _reward_process(self, last_action=None, legal_action=None, astar_action=None):
        """精简奖励函数：仅包含 A*引导、充电、取件、投递、步数惩罚、躲避无人机"""
        reward = 0.0

        # 1. 投递奖励（稀疏，高值）
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            reward += 1.0 * newly_delivered

        # 2. 步数惩罚（鼓励尽快完成任务）
        reward -= 0.002

    # 3. 非法动作惩罚（可选，如不需要可注释）
        if last_action is not None and legal_action is not None:
            if 0 <= last_action < len(legal_action) and legal_action[last_action] == 0:
                reward -= 0.2

    # 4. 停滞惩罚（已移除，仅保留 stuck_steps 计数供其他用途）
        if self.last_pos is not None:
            moved = abs(self.cur_pos[0] - self.last_pos[0]) + abs(self.cur_pos[1] - self.last_pos[1])
            if moved < 0.5:
                self.stuck_steps += 1
                # 不再给予惩罚
            else:
                self.stuck_steps -= 0.1
        self.last_pos = self.cur_pos

        # 5. A* 路径引导奖励（核心导航信号）
        if self.use_astar_reward and astar_action is not None and last_action is not None:
            if last_action == astar_action:
                reward += self.astar_reward_coef   # 跟随推荐动作
            else:
                reward -= 0.5                      # 偏离推荐动作
    #官方无人机惩罚
        # NPC_DANGER_DIST = 7.0
        # NPC_PENALTY = 5.0          # 固定惩罚值，可根据需要调整（例如 1.0 或 2.0）
        # for npc in self.npc_drones:
        #     npc_pos = (npc["pos"]["x"], npc["pos"]["z"])
        #     dist = np.hypot(self.cur_pos[0] - npc_pos[0], self.cur_pos[1] - npc_pos[1])
        #     if dist < NPC_DANGER_DIST:
        #         reward -= NPC_PENALTY
        #         break  # 一旦检测到进入危险距离，只惩罚一次，避免多个无人机重复惩罚（可选）

        # 官方无人机惩罚（指数型）
        NPC_DANGER_DIST = 7.0          # 危险距离阈值
        MAX_PENALTY = 10               # 最大惩罚（距离=0时）
        EXP_ALPHA = 0.3                 # 指数膨胀系数

        if self.npc_drones:
            # 计算当前最近距离
            min_dist = float('inf')
            for npc in self.npc_drones:
                npc_pos = (npc["pos"]["x"], npc["pos"]["z"])
                dist = np.hypot(self.cur_pos[0] - npc_pos[0], self.cur_pos[1] - npc_pos[1])
                if dist < min_dist:
                    min_dist = dist

            if min_dist < NPC_DANGER_DIST:
                # 初始化或更新上一步距离
                if not hasattr(self, 'last_min_npc_dist') or self.last_min_npc_dist is None:
                    self.last_min_npc_dist = min_dist
                else:
                    delta_dist = self.last_min_npc_dist - min_dist   # 正=远离，负=靠近
                    # 仅在靠近时惩罚
                    if delta_dist < 0:
                        # 指数惩罚公式：距离越近惩罚越大，边界处惩罚趋近0
                        # 归一化距离参数 t = (danger_dist - min_dist) / danger_dist  ∈ [0,1]
                        t = (NPC_DANGER_DIST - min_dist) / NPC_DANGER_DIST
                        # 指数型惩罚：-MAX_PENALTY * (exp(alpha * t) - 1) / (exp(alpha) - 1)
                        # 当 t=0 时惩罚为0，t=1 时惩罚为 -MAX_PENALTY
                        exp_term = np.exp(EXP_ALPHA * t)
                        penalty = -MAX_PENALTY * (exp_term - 1) / (np.exp(EXP_ALPHA) - 1)
                        reward += penalty
                    self.last_min_npc_dist = min_dist
            else:
                # 安全距离外，清除记忆
                self.last_min_npc_dist = None


        # 在官方无人机惩罚（指数型）代码块之后添加

        # ---- 新增：侧方 NPC 加速通过奖励 ----
        if self.npc_drones and astar_action is not None and last_action is not None:
            # 获取当前 A* 规划的下一步推荐动作
            # 该动作的方向向量（归一化）
            action_dirs = [
                (1,0), (0.707,-0.707), (0,-1), (-0.707,-0.707),
                (-1,0), (-0.707,0.707), (0,1), (0.707,0.707)
            ]
            rec_dir = action_dirs[astar_action]
            # 计算当前最近 NPC 的方向（从 agent 指向 NPC）
            min_dist = float('inf')
            npc_dir = (0,0)
            for npc in self.npc_drones:
                npc_pos = (npc["pos"]["x"], npc["pos"]["z"])
                dx = npc_pos[0] - self.cur_pos[0]
                dz = npc_pos[1] - self.cur_pos[1]
                dist = np.hypot(dx, dz)
                if dist < min_dist and dist > 1e-4:
                    min_dist = dist
                    npc_dir = (dx/dist, dz/dist)
            if min_dist < 7.0:   # 危险距离内
                # 计算推荐方向与 NPC 方向之间的夹角余弦
                dot = rec_dir[0]*npc_dir[0] + rec_dir[1]*npc_dir[1]
                # 如果 NPC 在路径的侧方（点积接近 0），且当前动作就是推荐动作
                if abs(dot) < 0.3 and last_action == astar_action:
                    # 给予加速通过奖励，距离越近奖励越大（鼓励快速离开）
                    reward += 0.2 * (1.0 - min_dist/7.0)
                # 7. 取件奖励（拿到新包裹时）
                new_packages = max(0, self.package_count - self.last_package_count)
                if new_packages > 0:
                    reward += 0.5 * new_packages
                self.last_package_count = self.package_count

        # 8. 充电奖励（仅低电量时靠近充电桩/仓库）
        LOW_BATTERY_THRESHOLD = 0.4   # 电量低于40%时考虑充电
        battery_ratio = self.battery / max(self.battery_max, 1)
        if battery_ratio < LOW_BATTERY_THRESHOLD:
            # 寻找最近的充电点（充电桩或仓库）
            min_charge_dist = float('inf')
            for cs in self.charging_stations:
                cs_pos = (cs["pos"]["x"], cs["pos"]["z"])
                dist = np.hypot(self.cur_pos[0] - cs_pos[0], self.cur_pos[1] - cs_pos[1])
                if dist < min_charge_dist:
                    min_charge_dist = dist
            if self.warehouse_pos is not None:
                dist_wh = np.hypot(self.cur_pos[0] - self.warehouse_pos[0],
                               self.cur_pos[1] - self.warehouse_pos[1])
                if dist_wh < min_charge_dist:
                    min_charge_dist = dist_wh

            CHARGE_REWARD_DIST = 10.0   # 在10格内开始给奖励
            if min_charge_dist < CHARGE_REWARD_DIST:
            # 距离越近奖励越高，最大0.5
                charge_reward = 1.0 * (1.0 - min_charge_dist / CHARGE_REWARD_DIST)
                reward += charge_reward

        # 所有其他奖励（徘徊、接近目标、进出仓库等）均已移除

        return [reward]