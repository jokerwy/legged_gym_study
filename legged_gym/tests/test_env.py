# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# =====================================================================
# 【模块说明】本文件是 legged_gym 的最小功能（冒烟）测试脚本：根据命令行指定的任务名
# 创建对应的仿真环境，用全零动作连续步进约 10 个 episode 长度，验证 Isaac Gym 环境能否
# 正常创建与运行（不加载/不训练策略）。用法示例：python test_env.py --task a1 --headless
# =====================================================================

import numpy as np
import os
from datetime import datetime

# 导入 Isaac Gym 主模块（必须在使用其任何 API 之前完成导入）
import isaacgym
# 导入 legged_gym.envs 包的全部内容（会触发 __init__.py 中的任务注册逻辑）
from legged_gym.envs import *
# 导入工具函数：命令行参数解析 get_args、策略导出 export_policy_as_jit、任务注册表 task_registry、日志器 Logger
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch


# test_env：环境冒烟测试函数
# 参数：args 命令行参数对象（由 get_args 解析，至少包含 task 任务名等字段）
# 功能：创建指定任务的环境（环境数限制到 10 个以内），用全零动作步进约 10 个 episode 后结束
def test_env(args):
    # 按任务名从任务注册表获取环境配置 env_cfg 与 PPO 训练配置 train_cfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # 测试时覆盖部分参数：把并行环境数量限制到最多 10 个，避免占用过多显存/算力
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 10)

    # prepare environment
    # 按任务名创建仿真环境实例（第二个返回值为附加信息，这里用不到，用 _ 丢弃）
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 主循环：连续步进 10 倍 episode 最大步数（约等于完整跑 10 个 episode）
    for i in range(int(10*env.max_episode_length)):
        # 构造全零动作张量 (num_envs, num_actions)：机器人保持在默认站姿（目标角=默认关节角）
        actions = 0.*torch.ones(env.num_envs, env.num_actions, device=env.device)
        # 执行一步仿真：返回观测 obs、特权观测（用 _ 丢弃）、奖励 rew、结束标志 done、附加信息 info
        obs, _, rew, done, info = env.step(actions)
    # 全部步进完成后打印 Done，表示环境创建与仿真循环运行正常
    print("Done")

# 脚本入口：解析命令行参数并运行测试
if __name__ == '__main__':
    # 解析命令行参数（如 --task a1、--headless、--sim_device cuda:0 等）
    args = get_args()
    # 调用冒烟测试函数
    test_env(args)
