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
# 【模块说明】本文件是 legged_gym.envs 包的入口（__init__.py），负责导入所有机器人
# 环境类及其配置类，并通过 task_registry 把每个任务名（如 "a1"、"cassie"）与对应的
# 环境类、环境配置实例、PPO 训练配置实例绑定注册，供训练/测试脚本按任务名创建环境。
# =====================================================================

# 从 legged_gym 包根模块导入项目根目录与环境目录两个路径常量
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
# 导入宇树 A1 机器人的环境配置类与 PPO 训练配置类
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
# 导入通用腿足式机器人环境基类（实现基于 Isaac Gym 的仿真循环、奖励计算等核心逻辑）
from .base.legged_robot import LeggedRobot
# 导入 ANYmal 系列机器人专用环境类（在基类之上支持执行器网络 actuator network）
from .anymal_c.anymal import Anymal
# 导入 ANYmal C 崎岖混合地形（mixed_terrains）任务的配置类与 PPO 配置类
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
# 导入 ANYmal C 平地（flat）任务的配置类与 PPO 配置类
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
# 导入 ANYmal B 机器人任务的配置类与 PPO 配置类
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
# 导入 Cassie 双足机器人专用环境类
from .cassie.cassie import Cassie
# 导入 Cassie 机器人任务的配置类与 PPO 配置类
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
# 再次导入 A1 配置类（与上文重复导入，功能等价，保持原样不改动）
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO


import os

# 导入任务注册表：维护"任务名 -> (环境类, 环境配置, 训练配置)"的全局映射
from legged_gym.utils.task_registry import task_registry

# 注册任务 "anymal_c_rough"：ANYmal C 机器人崎岖地形任务，使用 Anymal 专用环境类
task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
# 注册任务 "anymal_c_flat"：ANYmal C 机器人平地任务，使用 Anymal 专用环境类
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
# 注册任务 "anymal_b"：ANYmal B 机器人崎岖地形任务，复用 Anymal 专用环境类
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
# 注册任务 "a1"：宇树 A1 四足机器人任务，直接使用通用基类 LeggedRobot（无需专用环境类）
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
# 注册任务 "cassie"：Cassie 双足机器人任务，使用 Cassie 专用环境类
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
