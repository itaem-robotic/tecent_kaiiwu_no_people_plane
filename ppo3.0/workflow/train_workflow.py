#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery training workflow.
智运无人机训练工作流。
"""


import os
import time

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from typing import List, Optional, Tuple
from agent_ppo.feature.definition import ObsData, SampleData

class ParallelRunner:
    """Parallel runner for multiple environments.

    同时步进多个环境，收集样本并逐个 episode 返回。
    """

    def __init__(self, envs, agents, usr_conf, logger, monitor):
        self.envs = envs
        self.agents = agents
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.num_envs = len(envs)

        # 每个环境的当前状态
        self.obs_data_list = [None] * self.num_envs
        self.remain_info_list = [None] * self.num_envs
        self.done_list = [True] * self.num_envs   # True 表示需要 reset
        self.episode_counts = [0] * self.num_envs
        self.obs_data_list: List[Optional[ObsData]] = [None] * self.num_envs
        self.remain_info_list: List[Optional[dict]] = [None] * self.num_envs


    def run_episodes(self):
        """Generator that yields processed sample lists from any finished episode.

        每当一个环境完成一局，就对该局的 collector 做 sample_process 并 yield。
        """
        # 每个环境独立的样本收集器
        collectors = [[] for _ in range(self.num_envs)]
        episode_steps = [0] * self.num_envs
        episode_rewards = [0.0] * self.num_envs

        while True:
            # 1. Reset 所有已完成的环境
            for i in range(self.num_envs):
                if self.done_list[i]:
                    env_obs = self.envs[i].reset(self.usr_conf)
                    self.agents[i].reset(env_obs)
                    obs_data, remain_info = self.agents[i].observation_process(env_obs)
                    self.obs_data_list[i] = obs_data
                    self.remain_info_list[i] = remain_info
                    self.done_list[i] = False
                    episode_steps[i] = 0
                    episode_rewards[i] = 0.0
                    self.episode_counts[i] += 1

            # 2. 批量推理（每个环境独立调用 predict）
            act_data_list = []
            for i in range(self.num_envs):
                if not self.done_list[i]:
                    act_data = self.agents[i].predict([self.obs_data_list[i]])[0]
                    act_data_list.append(act_data)
                else:
                    act_data_list.append(None)

            # 3. 批量执行环境 step
            for i in range(self.num_envs):
                if self.done_list[i]:
                    continue

                # 执行动作
                act = self.agents[i].action_process(act_data_list[i])
                env_reward, env_obs = self.envs[i].step(act)

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                episode_steps[i] += 1
                done = terminated or truncated

                # 获取下一步观测和奖励（奖励已由 preprocessor 计算好）
                _obs_data, _remain_info = self.agents[i].observation_process(env_obs)
                reward = np.array(_remain_info["reward"], dtype=np.float32)   # 注意：remain_info 包含 reward
                episode_rewards[i] += float(reward.sum())

                # 构建 SampleData
                frame = SampleData(
                    obs=np.array(self.obs_data_list[i].feature, dtype=np.float32),
                    legal_action=np.array(self.obs_data_list[i].legal_action, dtype=np.float32),
                    act=np.array([act_data_list[i].action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    value=act_data_list[i].value.flatten()[:Config.VALUE_NUM],
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data_list[i].prob, dtype=np.float32),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                )
                collectors[i].append(frame)

                # 如果 episode 结束，处理 final_reward，并 yield 样本
                if done:
                    # 添加终止惩罚
                    final_reward = np.zeros(Config.VALUE_NUM, dtype=np.float32)
                    if terminated:
                        final_reward[0] = -1.0
                    collectors[i][-1].reward = collectors[i][-1].reward + final_reward

                    # 可选：上报监控
                    if self.monitor:
                        self.monitor.put_data({
                            os.getpid(): {
                                "reward": round(episode_rewards[i], 4),
                                "episode_cnt": self.episode_counts[i],
                                #"episode_cnt": self.agents[i].episode_cnt,   # 需要 agent 有 episode_cnt，或自己维护
                                "delivered": self.agents[i].preprocessor.delivered,
                            }
                        })

                    # 处理样本（GAE + 可选归一化）
                    if len(collectors[i]) > 0:
                        processed_samples = sample_process(collectors[i])
                        yield processed_samples   # 返回这一局的样本列表

                    # 清空该环境的 collector，标记 done 以便下次 reset
                    collectors[i] = []
                    self.done_list[i] = True
                else:
                    # 更新当前状态
                    self.obs_data_list[i] = _obs_data
                    self.remain_info_list[i] = _remain_info

# def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
#     """Training entry point called by the platform.

#     训练入口，平台调用。
#     """
#     last_save_model_time = time.time()
#     env = envs[0]
#     agent = agents[0]

#     usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
#     if usr_conf is None:
#         logger.error("usr_conf is None, check agent_ppo/conf/train_env_conf.toml")
#         return

#     episode_runner = EpisodeRunner(
#         env=env,
#         agent=agent,
#         usr_conf=usr_conf,
#         logger=logger,
#         monitor=monitor,
#     )

#     while True:
#         for g_data in episode_runner.run_episodes():
#             agent.send_sample_data(g_data)
#             g_data.clear()

#             now = time.time()
#             if now - last_save_model_time >= 600:
#                 agent.save_model()
#                 last_save_model_time = now

def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Training entry point with parallel environments."""
    last_save_model_time = time.time()
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, check agent_ppo/conf/train_env_conf.toml")
        return

    # 创建并行运行器
    runner = ParallelRunner(envs, agents, usr_conf, logger, monitor)

    # 主循环：从 runner 中不断获取已完成 episode 的样本，并发送给训练
    for sample_data in runner.run_episodes():
        # 发送样本（使用任意一个 agent，所有 agent 共享训练通道）
        agents[0].send_sample_data(sample_data)
        sample_data.clear()   # 释放内存

        # 定期保存模型（每 10 分钟）
        now = time.time()
        if now - last_save_model_time >= 600:
            agents[0].save_model()
            last_save_model_time = now

# class EpisodeRunner:
#     """Single-episode runner.

#     单局运行器。
#     """

#     def __init__(self, env, agent, usr_conf, logger, monitor):
#         self.env = env
#         self.agent = agent
#         self.usr_conf = usr_conf
#         self.logger = logger
#         self.monitor = monitor
#         self.episode_cnt = 0
#         self.last_report_monitor_time = 0
#         self.last_training_metrics_time = 0

#     def run_episodes(self):
#         """Run one episode and yield the sample list.

#         运行一局并 yield 样本列表。
#         """
#         while True:
#             # Print training metrics periodically / 定期打印训练指标
#             now = time.time()
#             if now - self.last_training_metrics_time >= 60:
#                 metrics = get_training_metrics()
#                 self.last_training_metrics_time = now
#                 if metrics:
#                     self.logger.info(f"training_metrics: {metrics}")

#             # Reset environment / 重置环境
#             env_obs = self.env.reset(self.usr_conf)
#             if handle_disaster_recovery(env_obs, self.logger):
#                 continue

#             self.agent.reset(env_obs)
#             #self.agent.load_model(id="latest")

#             obs_data, remain_info = self.agent.observation_process(env_obs)

#             collector = []
#             self.episode_cnt += 1
#             done = False
#             step = 0
#             total_delivered = 0
#             total_reward_sum = 0.0

#             self.logger.info(f"Episode {self.episode_cnt} start, conf={self.usr_conf}")

#             # Main loop / 主循环
#             while not done:
#                 # Inference / 推理
#                 act_data = self.agent.predict(list_obs_data=[obs_data])[0]
#                 act = self.agent.action_process(act_data)

#                 # Environment step / 环境交互
#                 env_reward, env_obs = self.env.step(act)
#                 if handle_disaster_recovery(env_obs, self.logger):
#                     break

#                 terminated = env_obs["terminated"]
#                 truncated = env_obs["truncated"]
#                 step += 1
#                 done = terminated or truncated

#                 # Next step observation / 下一步观测
#                 _obs_data, _remain_info = self.agent.observation_process(env_obs)

#                 # Reward (from preprocessor, single channel) / 奖励（来自 preprocessor，单通道 list[float]）
#                 reward = np.array(self.agent.preprocessor._reward_process(), dtype=np.float32)

#                 total_reward_sum += float(reward.sum())

#                 # End-of-episode additional reward / 局末额外奖励
#                 final_reward = np.zeros(Config.VALUE_NUM, dtype=np.float32)
#                 if done:
#                     total_delivered = self.agent.preprocessor.delivered
#                     total_score = env_obs["observation"]["env_info"].get("total_score", 0)

#                     if terminated:
#                         # Abnormal termination (collision or energy depleted): small penalty
#                         # 异常终止（碰撞 or 电量耗尽）：给小惩罚
#                         final_reward[0] = -1.0
#                         result_str = "FAIL"
#                     else:
#                         # Normal end: reached max steps
#                         # 正常到达最大步数
#                         final_reward[0] = 0.0
#                         result_str = "WIN"

#                     self.logger.info(
#                         f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
#                         f"result:{result_str} delivered:{total_delivered} "
#                         f"score:{total_score} total_reward:{total_reward_sum:.2f}"
#                     )

#                 # Build sample frame / 构造样本帧
#                 obs_data = self.obs_data_list[i]
#                 assert obs_data is not None, "obs_data should not be None"
#                 frame = SampleData(
#                     obs=np.array(obs_data.feature, dtype=np.float32),
#                     legal_action=np.array(obs_data.legal_action, dtype=np.float32),
#                     act=np.array([act_data.action[0]], dtype=np.float32),
#                     reward=reward,
#                     done=np.array([float(done)], dtype=np.float32),
#                     value=act_data.value.flatten()[: Config.VALUE_NUM],
#                     next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
#                     advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
#                     prob=np.array(act_data.prob, dtype=np.float32),
#                     reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
#                 )
#                 collector.append(frame)

#                 # End of episode: merge final_reward, compute GAE, yield
#                 # 局末：合并 final_reward、GAE、yield
#                 if done:
#                     if len(collector) > 0:
#                         collector[-1].reward = collector[-1].reward + final_reward

#                     now = time.time()
#                     if now - self.last_report_monitor_time >= 60 and self.monitor:
#                         self.monitor.put_data(
#                             {
#                                 os.getpid(): {
#                                     "reward": round(total_reward_sum + final_reward[0], 4),
#                                     "episode_cnt": self.episode_cnt,
#                                     "delivered": total_delivered,
#                                 }
#                             }
#                         )
#                         self.last_report_monitor_time = now

#                     if len(collector) > 0:
#                         collector = sample_process(collector)
#                         yield collector
#                     break

#                 obs_data = _obs_data
#                 remain_info = _remain_info
