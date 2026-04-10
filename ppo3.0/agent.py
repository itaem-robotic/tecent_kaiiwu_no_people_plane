#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery Agent class. Inherits BaseAgent, implements PPO inference, training, and model save/load.
智运无人机 Agent 主类。继承 BaseAgent，实现 PPO 推理、训练、存取模型。
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.lr_step_count = 0          # 记录衰减步数
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        super().__init__(agent_type, device, logger, monitor)
        self.current_stuck_flag = False

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def _forward(self, feature, legal_action):
        """Gradient-free forward pass, returns (logits, value).

        无梯度前向推理，返回 (logits, value)。
        """
        self.model.set_eval_mode()
        obs_t = (
            torch.tensor(np.array([feature]), dtype=torch.float32).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        with torch.no_grad():
            logits, value = self.model(obs_t, inference=True)
        return logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def _get_min_npc_distance(self):
        if not self.npc_drones:
            return 999.0
        min_dist = min(np.hypot(self.cur_pos[0]-npc["pos"]["x"], self.cur_pos[1]-npc["pos"]["z"]) for npc in self.npc_drones)
        return min_dist


    def predict(self, list_obs_data):
        """Training inference: sample action from probability distribution.

        训练时推理：按概率采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
            # ========== 新增：紧急避障（硬编码） ==========
        if hasattr(self, 'preprocessor'):
            danger_dist = 3.5  # 危险距离阈值（格）
            min_npc_dist = self.preprocessor._get_min_npc_distance()
            if min_npc_dist < danger_dist:
                escape_act = self.preprocessor.get_escape_action()
                if escape_act is not None and legal_action[escape_act] == 1:
                    # 构建确定性动作输出
                    prob = [0.0] * len(legal_action)
                    prob[escape_act] = 1.0
                    # value 可以沿用当前网络预测（可选）或设为 0，这里简化为 0
                    return [ActData(action=[escape_act], d_action=[escape_act], prob=prob, value=np.array([0.0]))]
        # ==========================================
            logits, value = self._forward(feature, legal_action)
          # ---------- 卡住脱困噪声 ----------
        if self.current_stuck_flag:
            # 可选：根据 stuck_steps 动态调整噪声强度
            #   noise_std = min(0.1 * self.current_stuck_steps, 1.0)
            noise_std = 0.5
            noise = np.random.randn(*logits.shape) * noise_std
            logits = logits + noise

        legal_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_np)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)
        return [ActData(action=[action], d_action=[d_action], prob=list(prob), value=np.array([value]))]
    def exploit(self, env_obs):
        """Evaluation inference: greedy action selection.

        评估时推理：贪心选最大概率动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        if obs_data is None:
            return 0
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update.

        委托给 Algorithm 执行 PPO 更新。
        """
        # result = self.algorithm.learn(list_sample_data)
        # self.lr_step_count += 1
        # if self.lr_step_count % Config.LR_DECAY_STEPS == 0:
        #     for param_group in self.optimizer.param_groups:
        #         new_lr = max(param_group['lr'] * Config.LR_DECAY_RATE, Config.LR_MIN)
        #         param_group['lr'] = new_lr
        #     if self.logger:
        #         self.logger.info(f"Learning rate decayed to {new_lr:.6f}")

        # return result
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, env_obs):
        """Convert raw env observation to ObsData + remain_info.

        将原始环境观测转换为 ObsData + remain_info。
        """
        feature, legal_action, reward, stuck_flag  = self.preprocessor.feature_process(env_obs, self.last_action)
        self.current_stuck_flag = stuck_flag
        remain_info = {"reward": reward, "stuck_flag": stuck_flag}
        return (
            ObsData(feature=list(feature), legal_action=legal_action),
            remain_info,
        )

    def action_process(self, act_data, is_stochastic=True):
        """Extract int action from ActData and update last_action.

        从 ActData 提取整数动作，并更新 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def _legal_soft_max(self, logits, legal_action):
        """Apply legal action mask and compute normalized probabilities.

        对 logits 应用合法动作掩码并计算归一化概率。
        """
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample from probability distribution (or argmax).

        从概率分布中采样（或取最大值）。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
