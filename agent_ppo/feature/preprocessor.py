#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Blank PPO.
空白版 PPO 特征预处理与奖励设计。
"""

import numpy as np
from agent_ppo.conf.conf import Config


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.step_no = 0

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        self.step_no += 1

        # Blank feature: constant zero / 空白特征：常量零
        feature = np.zeros(Config.DIM_OF_OBSERVATION, dtype=np.float32)

        # All actions legal / 所有动作合法
        legal_action = [1] * Config.ACTION_NUM

        # Zero reward / 零奖励
        reward = [0.0]

        return feature, legal_action, reward
