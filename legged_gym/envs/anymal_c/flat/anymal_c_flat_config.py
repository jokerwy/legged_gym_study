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
# 【模块说明】本文件定义 ANYmal C 四足机器人在平地（Flat Terrain）任务下的配置。
# AnymalCFlatCfg 继承自崎岖地形配置 AnymalCRoughCfg，仅覆盖与平地任务不同的部分
# （关闭地形高度测量、调整奖励权重、修改指令范围等）；AnymalCFlatCfgPPO 为 PPO 训练配置。
# =====================================================================

# 从 legged_gym.envs 包入口导入 ANYmal C 崎岖地形配置类与 PPO 配置类（作为父类复用）
from legged_gym.envs import AnymalCRoughCfg, AnymalCRoughCfgPPO

# AnymalCFlatCfg：ANYmal C 平地任务的环境配置，继承自 AnymalCRoughCfg（崎岖地形配置）
class AnymalCFlatCfg( AnymalCRoughCfg ):
    # env：环境规模与观测维度配置（平地不使用地形高度测量，因此观测维度更小）
    class env( AnymalCRoughCfg.env ):
        # 观测向量维度：48 维（不含崎岖地形中额外的地形高度扫描点观测）
        num_observations = 48
  
    # terrain：地形配置，平地任务使用无限平面
    class terrain( AnymalCRoughCfg.terrain ):
        # 地形网格类型：'plane' 表示无限平面（无起伏）
        mesh_type = 'plane'
        # 是否测量机身周围的地形高度：平地无需测量，关闭以节省观测维度与计算量
        measure_heights = False
  
    # asset：模型资源配置
    class asset( AnymalCRoughCfg.asset ):
        # 自碰撞开关：0 表示启用机器人自身各部件间的碰撞检测（平地任务开启）
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    # rewards：奖励函数配置
    class rewards( AnymalCRoughCfg.rewards ):
        # 足端接触力上限（牛）：平地任务设为 350N，超过会被 feet_contact_forces 项惩罚
        max_contact_force = 350.
        # scales：各奖励项权重，在父类（崎岖地形）基础上覆盖以下几项
        class scales ( AnymalCRoughCfg.rewards.scales ):
            # 机身姿态（横滚/俯仰偏离重力方向）惩罚权重：平地任务加大到 -5.0，要求机身更平稳
            orientation = -5.0
            # 关节力矩惩罚权重（比崎岖地形更小，平地能耗约束相对宽松）
            torques = -0.000025
            # 单脚腾空时间奖励权重：鼓励脚有一定的摆动腾空相
            feet_air_time = 2.
            # feet_contact_forces = -0.01
    
    # commands：速度指令生成配置
    class commands( AnymalCRoughCfg.commands ):
        # 是否使用航向角指令模式：False 表示直接给定期望偏航角速度，而非给定期望航向角
        heading_command = False
        # 指令重采样时间间隔（秒）：每隔 4 秒随机采样一次新的速度指令
        resampling_time = 4.
        # ranges：各指令分量的随机采样范围
        class ranges( AnymalCRoughCfg.commands.ranges ):
            # 偏航角速度指令的采样范围 [最小值, 最大值]，单位 rad/s
            ang_vel_yaw = [-1.5, 1.5]

    # domain_rand：域随机化配置（提升策略鲁棒性，利于 sim-to-real 迁移）
    class domain_rand( AnymalCRoughCfg.domain_rand ):
        # 地面摩擦系数随机化范围：平地地面上采用平均组合模式，即总摩擦 = (足底摩擦 + 1.)/2
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

# AnymalCFlatCfgPPO：ANYmal C 平地任务的 PPO 训练配置，继承自 AnymalCRoughCfgPPO
class AnymalCFlatCfgPPO( AnymalCRoughCfgPPO ):
    # policy：策略网络（Actor）与价值网络（Critic）的结构配置
    class policy( AnymalCRoughCfgPPO.policy ):
        # Actor 网络隐藏层维度：较小的 3 层 MLP（平地任务观测少，网络可更小）
        actor_hidden_dims = [128, 64, 32]
        # Critic 网络隐藏层维度
        critic_hidden_dims = [128, 64, 32]
        # 激活函数类型
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    # algorithm：PPO 算法超参数
    class algorithm( AnymalCRoughCfgPPO.algorithm):
        # 熵系数：策略熵正则项权重，鼓励探索
        entropy_coef = 0.01

    # runner：训练运行器（实验管理）配置
    class runner ( AnymalCRoughCfgPPO.runner):
        # 本次运行名称（留空则由时间戳自动生成）
        run_name = ''
        # 实验名称：日志与模型保存目录以此命名
        experiment_name = 'flat_anymal_c'
        # 要加载的训练运行编号：-1 表示加载最近一次（最新）的运行
        load_run = -1
        # 最大训练迭代次数：平地任务较简单，300 次迭代通常即可收敛
        max_iterations = 300
