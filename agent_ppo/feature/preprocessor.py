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
    abs_norm = norm(np.array(target_pos), 128, -128)
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

    def update_from_local_map(self, center_gx, center_gz, local_map):
        half = len(local_map) // 2
        for dz in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gx = center_gx + dx
                gz = center_gz + dz
                if 0 <= gx < self.width and 0 <= gz < self.height:
                    self.known_grids.add((gx, gz))       # 标记已知
                    if local_map[dz+half][dx+half] == 0:
                        self.obstacles.add((gx, gz))
                    else:
                        self.obstacles.discard((gx, gz))

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
                if not self.is_valid(*neighbor):
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

#预处理器类，包含状态重置、观察解析、特征提取、合法动作获取和奖励计算等方法
class Preprocessor:
    """feature preprocessor for Drone Delivery.

    智运无人机预处理器，仅保留最少信息。
    """

    def __init__(self, use_astar_feature=True, use_astar_reward=True):
        self.reset()
        # 初始化 A* 规划器（地图边界根据环境设定，这里用 [-128, 128]）
        self.astar = AStarPlanner(grid_size=128)
        self.use_astar_feature = use_astar_feature
        self.use_astar_reward = use_astar_reward

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

        hero = frame_state["heroes"]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        self.battery = hero.get("battery", self.battery_max)
        self.battery_max = hero.get("battery_max", 100)
        self.packages = hero.get("packages", [])

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

        self.npc_drones = []
        for npc in frame_state.get("npcs", []):
            # 根据实际字段判断，此处假设所有 npcs 均为无人机
            self.npc_drones.append(npc)

        self.legal_act = obs.get("legal_action", [1] * 8)
        map_info = obs.get("map_info")
        if map_info is not None:
            gx, gz = self._world_to_grid(*self.cur_pos)
            self.astar.update_from_local_map(gx, gz, map_info)

    def feature_process(self, env_obs, last_action):
        """Core feature extraction. Returns (feature_22d, legal_action, reward).

        核心特征提取方法，返回 22 维特征向量、合法动作掩码和奖励。
        """
        self._parse_obs(env_obs)
        cur_pos = self.cur_pos   # 已经更新为最新位置
        # 1. Hero state features (4D) / 英雄状态特征（4D）
        battery_ratio = norm(self.battery, self.battery_max)
        package_count_norm = norm(len(self.packages), 3)
        cur_pos_norm = norm(np.array(self.cur_pos, dtype=float), 128, -128)
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
        target_ids = set(self.packages)
        # A* 推荐动作特征（8D one-hot）
        astar_action_feat = np.zeros(8, dtype=float)
        astar_recommended_action = None
        target_pos = self._get_current_target_position()
        if target_pos is not None:
            start_grid = self._world_to_grid(*cur_pos)
            goal_grid = self._world_to_grid(*target_pos)
            #astar_recommended_action = self.astar.next_action(start_grid, goal_grid)
            astar_recommended_action = None
            if target_pos is not None:
                start_grid = self._world_to_grid(*cur_pos)
                global_goal_grid = self._world_to_grid(*target_pos)
                # 如果全局目标就在局部视野内，直接用全局目标
                if max(abs(global_goal_grid[0]-start_grid[0]), abs(global_goal_grid[1]-start_grid[1])) <= 10:
                    local_goal = global_goal_grid
                else:
                    local_goal = self._get_local_subgoal(start_grid, global_goal_grid)
                if local_goal is not None:
                    astar_recommended_action = self.astar.next_action(start_grid, local_goal)
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

        astar_action_feat = np.zeros(8, dtype=float)
        astar_recommended_action = None
        target_pos = self._get_current_target_position()
        if target_pos is not None:
            start_grid = self._world_to_grid(*cur_pos)
            goal_grid = self._world_to_grid(*target_pos)
            astar_recommended_action = self.astar.next_action(start_grid, goal_grid)
            if astar_recommended_action is not None:
                astar_action_feat[astar_recommended_action] = 1.0   # one-hot 表示

        if self.use_astar_feature:
            feature = np.concatenate([
                hero_feat,
                station_feat,
                np.array(legal_action, dtype=float),
                indicators,
                astar_action_feat          # 新增 8 维 → 总维度变为 30
            ])
        else:
            feature = np.concatenate([
                hero_feat,
                station_feat,
                np.array(legal_action, dtype=float),
                indicators
            ])

        # 计算奖励（传入 last_action 和 astar 推荐动作）
        reward = self._reward_process(last_action, legal_action, astar_recommended_action)
        return feature, legal_action, reward



        # Concatenate features (Total 22D / 合计 22D)
        # feature = np.concatenate(
        #     [
        #         hero_feat,
        #         station_feat,
        #         np.array(legal_action, dtype=float),
        #         indicators,
        #     ]
        # )

        # #reward = self._reward_process()
        # reward = self._reward_process(last_action, legal_action)
        # return feature, legal_action, reward

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

    # def _reward_process(self):
    #     """Reward function.

    #     奖励函数。
    #     """
    #     reward = 0.0

    #     # 1. Delivery reward / 投递奖励
    #     newly_delivered = max(0, self.delivered - self.last_delivered)
    #     if newly_delivered > 0:
    #         reward += 1.0 * newly_delivered

    #     # 2. Step penalty / 步数惩罚
    #     reward -= 0.001

    #     return [reward]
    # ILLEGAL_ACTION_PENALTY = 1.0
    # def _reward_process(self, last_action=None, legal_action=None):
    #     """Reward function with illegal action penalty and dense guidance.
        
    #     奖励函数，包含非法动作惩罚、靠近目标奖励、停滞惩罚等。
    #     参数 last_action 和 legal_action 为可选，若传入则计算非法动作惩罚。
    #     """
    #     reward = 0.0
    #     if last_action is not None and legal_action is not None:
    #         if last_action >= 0 and last_action < len(legal_action):
    #             if legal_action[last_action] == 0:
    #                 reward -= self.ILLEGAL_ACTION_PENALTY
    #                 print(f"ILLEGAL action {last_action} penalized! reward={reward}")  # 临时日志
    #             else:
    #                 print(f"Legal action {last_action}")  # 可选

        

    #     # 1. Delivery reward
    #     newly_delivered = max(0, self.delivered - self.last_delivered)
    #     if newly_delivered > 0:
    #         reward += 1.0 * newly_delivered

    #     # 2. Step penalty
    #     reward -= 0.001

    #     # 3. Illegal action penalty (only if parameters provided)
    #     if last_action is not None and legal_action is not None:
    #         if last_action >= 0 and last_action < len(legal_action):
    #             if legal_action[last_action] == 0:
    #                 reward -= getattr(self, 'ILLEGAL_ACTION_PENALTY', 0.5)
    #         elif last_action >= 0:
    #             reward -= getattr(self, 'ILLEGAL_ACTION_PENALTY', 0.5)

    #     # 4. Stuck penalty (using self.cur_pos which is already updated)
    #     if hasattr(self, 'last_pos') and self.last_pos is not None:
    #         dx = self.cur_pos[0] - self.last_pos[0]
    #         dz = self.cur_pos[1] - self.last_pos[1]
    #         moved = abs(dx) + abs(dz)
    #         if moved < 0.5:
    #             self.stuck_steps = getattr(self, 'stuck_steps', 0) + 1
    #             reward -= 0.01 * self.stuck_steps
    #         else:
    #             self.stuck_steps = 0
    #     self.last_pos = self.cur_pos

    #     # 5. Dense reward for approaching target station
    #     target_pos = self._get_current_target_position()
    #     if target_pos is not None:
    #         cur_dist = np.sqrt((self.cur_pos[0]-target_pos[0])**2 + (self.cur_pos[1]-target_pos[1])**2)
    #         if hasattr(self, 'last_dist_to_target') and self.last_dist_to_target is not None:
    #             delta_dist = self.last_dist_to_target - cur_dist
    #             reward += delta_dist * 0.05
    #         self.last_dist_to_target = cur_dist
    #     else:
    #         self.last_dist_to_target = None

    #     return [reward]
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
        # 如果有携带包裹，目标为第一个包裹对应的驿站
        if self.packages:
            target_id = self.packages[0]
            for s in self.stations:
                if s.get("config_id") == target_id:
                    return (s["pos"]["x"], s["pos"]["z"])
            return None
        else:
            # 没有包裹时，可返回最近的驿站（取货点）或返回None
            # 这里复用原有排序逻辑
            if not self.stations:
                return None
            target_ids = set()
            def station_sort_key(s):
                is_tgt = s.get("config_id", 0) in target_ids
                dist = np.sqrt((s["pos"]["x"]-self.cur_pos[0])**2 + (s["pos"]["z"]-self.cur_pos[1])**2)
                return (0 if is_tgt else 1, dist)
            sorted_stations = sorted(self.stations, key=station_sort_key)
            if sorted_stations:
                return (sorted_stations[0]["pos"]["x"], sorted_stations[0]["pos"]["z"])
            return None


    def _reward_process(self, last_action=None, legal_action=None, astar_action=None):
        """奖励函数，支持非法动作惩罚 + 停滞惩罚 + 接近目标奖励 + A*引导奖励"""
        reward = 0.0

        # 1. 投递奖励
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            reward += 1.0 * newly_delivered

        # 2. 步数惩罚
        reward -= 0.001

        # 3. 非法动作惩罚
        if last_action is not None and legal_action is not None:
            if 0 <= last_action < len(legal_action) and legal_action[last_action] == 0:
                reward -= 0.5

        # 4. 停滞惩罚（复用原有逻辑）
        if self.last_pos is not None:
            moved = abs(self.cur_pos[0] - self.last_pos[0]) + abs(self.cur_pos[1] - self.last_pos[1])
            if moved < 0.5:
                self.stuck_steps += 1
                reward -= 0.01 * self.stuck_steps
            else:
                self.stuck_steps = 0
        self.last_pos = self.cur_pos

        # 5. 接近目标奖励（密集）
        target_pos = self._get_current_target_position()
        if target_pos is not None:
            cur_dist = np.hypot(self.cur_pos[0]-target_pos[0], self.cur_pos[1]-target_pos[1])
            if self.last_dist_to_target is not None:
                delta_dist = self.last_dist_to_target - cur_dist
                reward += delta_dist * 0.05
            self.last_dist_to_target = cur_dist
        else:
            self.last_dist_to_target = None

        # 6. 新增：A* 路径引导奖励（鼓励执行推荐动作）
        if self.use_astar_reward and astar_action is not None and last_action is not None:
            if last_action == astar_action:
                reward += 0.02   # 跟随 A* 建议给予小额奖励
            else:
                reward -= 0.01   # 偏离 A* 路径轻微惩罚

        # 7. 靠近官方无人机惩罚（距离越近惩罚越大）
        NPC_DIST_THRESHOLD = 2.5
        NPC_PENALTY_MAX = 0.3
        for npc in self.npc_drones:
            npc_pos = (npc["pos"]["x"], npc["pos"]["z"])
            dist = np.hypot(self.cur_pos[0] - npc_pos[0], self.cur_pos[1] - npc_pos[1])
            if dist < NPC_DIST_THRESHOLD:
                penalty = (1.0 - dist / NPC_DIST_THRESHOLD) * NPC_PENALTY_MAX
                reward -= penalty

        # 8. 原地不动额外惩罚（连续停滞超过3步）
        if self.stuck_steps >= 3:
            reward -= 0.1

        # 9. 充电桩附近徘徊惩罚
        CHARGE_LOITER_LIMIT = 10
        near_charge = False
        for cs in self.charging_stations:
            cs_pos = (cs["pos"]["x"], cs["pos"]["z"])
            charge_range = cs.get("range", 5.0)   # 使用充电桩自带的范围字段
            if np.hypot(self.cur_pos[0] - cs_pos[0], self.cur_pos[1] - cs_pos[1]) < charge_range:
                near_charge = True
                break

        if near_charge:
            self.charge_loiter_steps += 1
            if self.charge_loiter_steps > CHARGE_LOITER_LIMIT:
                reward -= 0.02
        else:
            self.charge_loiter_steps = 0

        # 10. 鼓励取包裹（无包裹时向驿站移动的额外奖励）
        if not self.packages and target_pos is not None:
            reward += 0.005
        return [reward]