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
# 【模块说明】本文件是 legged_gym 的【训练入口脚本】。
# 整体训练流程非常简洁，只有三步：
#   1) task_registry.make_env()        —— 按任务名创建 Isaac Gym 仿真环境；
#   2) task_registry.make_alg_runner() —— 创建 PPO 训练 runner（含网络与算法）；
#   3) ppo_runner.learn()              —— 启动 PPO 训练主循环。
# 典型用法：python train.py --task=anymal_c_flat --headless --max_iterations=1000
# =====================================================================
import numpy as np
import os
from datetime import datetime

# 中文：必须先 import isaacgym，再 import 任何 torch / legged_gym 模块
# （Isaac Gym 要求在所有依赖它的库之前完成初始化，顺序错误会报错）
import isaacgym
# 中文：导入 legged_gym.envs 包——其 __init__.py 会在导入时把所有任务
# （如 anymal_c_flat、anymal_c_rough、cassie 等）注册到全局 task_registry
from legged_gym.envs import *
# 中文：导入命令行参数解析函数与全局任务注册表单例
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    # 中文：训练主函数。参数 args 为 get_args() 解析出的命令行参数命名空间，
    # 关键字段：args.task（任务名）、args.num_envs（并行环境数）、
    # args.seed（随机种子）、args.headless（无界面）等。
    # 中文：第 1 步——创建仿真环境：按任务名查找已注册的环境类与配置，
    # 固定随机种子、解析仿真参数，实例化向量化环境（内部创建 N 个并行机器人）。
    # 返回：env - 环境实例；env_cfg - 最终生效的环境配置。
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # 中文：第 2 步——创建 PPO 训练 runner：加载训练配置、生成带时间戳的日志目录
    # （logs/<experiment_name>/<时间戳>_<run_name>），实例化 rsl_rl 的
    # OnPolicyRunner（内部创建 ActorCritic 网络与 PPO 算法）；
    # 若 --resume 则同时加载已有模型权重继续训练。
    # 返回：ppo_runner - 训练执行器；train_cfg - 最终生效的训练配置。
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # 中文：第 3 步——启动 PPO 训练主循环。
    # num_learning_iterations：总迭代轮数（每轮 = 采集一批数据 + 若干次策略更新），
    # 取自 train_cfg.runner.max_iterations（可被 --max_iterations 覆盖）；
    # init_at_random_ep_len=True：首次训练时给各环境随机初始化回合计数，
    # 使所有环境的回合结束时刻错开，避免所有机器人同时 reset 导致数据强相关。
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    # 中文：脚本入口：先解析命令行参数，再执行训练
    args = get_args()
    train(args)
