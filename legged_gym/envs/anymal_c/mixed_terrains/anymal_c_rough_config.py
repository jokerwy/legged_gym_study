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
# 【模块说明】本文件定义 ANYmal C 四足机器人在崎岖混合地形（Mixed Terrains）任务下的配置：
# AnymalCRoughCfg 为环境配置（并行环境数、地形、初始状态、控制参数、域随机化、奖励），
# AnymalCRoughCfgPPO 为 PPO 训练配置。本文件还是 ANYmal B 配置与 ANYmal C 平地配置的父类。
# =====================================================================

# 从基础配置模块导入通用腿足机器人环境配置基类与 PPO 训练配置基类
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# AnymalCRoughCfg：ANYmal C 机器人崎岖地形任务的环境配置类，继承自通用配置 LeggedRobotCfg
class AnymalCRoughCfg( LeggedRobotCfg ):
    # env：并行环境规模与动作维度配置
    class env( LeggedRobotCfg.env ):
        # 并行仿真的环境数量（GPU 大规模并行采样）
        num_envs = 4096
        # 动作维度：ANYmal C 共 12 个驱动关节（4 条腿 × 每腿 HAA/HFE/KFE 3 个关节）
        num_actions = 12

    # terrain：地形配置
    class terrain( LeggedRobotCfg.terrain ):
        # 地形网格类型：'trimesh' 表示由三角网格构成的程序化崎岖地形（含坡道、台阶、离散障碍等）
        mesh_type = 'trimesh'

    # init_state：机器人初始状态配置（基座初始位置与各关节默认角度）
    class init_state( LeggedRobotCfg.init_state ):
        # 基座初始位置 [x, y, z]，单位米；ANYmal C 站立时机身高度约 0.6 米
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        # 各关节默认角度（弧度），即策略输出 action = 0.0 时的目标关节角（机器人标准站姿）
        # 命名规则：LF/LH/RF/RH = 左前/左后/右前/右后 腿；HAA=髋侧摆关节、HFE=髋屈伸关节、KFE=膝关节
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 四条腿的 HAA（髋侧摆）关节默认角
            "LF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RF_HAA": -0.0,
            "RH_HAA": -0.0,

            # 四条腿的 HFE（髋屈伸）关节默认角：前腿为正、后腿为负
            "LF_HFE": 0.4,
            "LH_HFE": -0.4,
            "RF_HFE": 0.4,
            "RH_HFE": -0.4,

            # 四条腿的 KFE（膝）关节默认角：前腿为负、后腿为正（站姿弯曲方向相反）
            "LF_KFE": -0.8,
            "LH_KFE": 0.8,
            "RF_KFE": -0.8,
            "RH_KFE": 0.8,
        }

    # control：关节控制参数（ANYmal C 默认使用执行器网络，PD 增益主要供执行器网络内部参考）
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # 各关节 PD 刚度（比例增益 Kp），按关节类型 HAA/HFE/KFE 分别设置，单位 N*m/rad
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        # 各关节 PD 阻尼（微分增益 Kd），单位 N*m*s/rad
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # 动作缩放系数：目标关节角 = 策略输出动作 * action_scale + 默认关节角
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        # 降采样系数：每个策略步（policy DT）内执行 4 次仿真+控制步（sim DT）
        decimation = 4
        # 是否使用执行器网络：True 表示用 LSTM 执行器模型（模拟 ANYDrive 电机动力学）代替理想 PD 控制器
        use_actuator_network = True
        # 预训练执行器网络模型文件路径（TorchScript 格式的 anydrive_v3 LSTM 模型）
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    # asset：机器人模型资源配置
    class asset( LeggedRobotCfg.asset ):
        # ANYmal C 机器人 URDF 模型文件路径
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
        # 机器人资产名称
        name = "anymal_c"
        # 足底刚体名称（ANYmal 系列统一命名为 FOOT）
        foot_name = "FOOT"
        # 小腿（SHANK）与大腿（THIGH）部位接触地面时给予惩罚的刚体名
        penalize_contacts_on = ["SHANK", "THIGH"]
        # 基座（base）接触地面时直接终止当前 episode（视为摔倒）
        terminate_after_contacts_on = ["base"]
        # 自碰撞开关：1 表示禁用自身部件间碰撞检测，0 表示启用
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    # domain_rand：域随机化配置（提升策略鲁棒性）
    class domain_rand( LeggedRobotCfg.domain_rand):
        # 是否随机化基座质量
        randomize_base_mass = True
        # 基座附加质量的随机范围（千克）：在 [-5, 5] kg 内随机增减，模拟负载变化
        added_mass_range = [-5., 5.]
  
    # rewards：奖励函数配置
    class rewards( LeggedRobotCfg.rewards ):
        # 机身目标高度（米）：用于 base_height 奖励项，ANYmal C 站姿机身高约 0.5 米
        base_height_target = 0.5
        # 足端接触力上限（牛）：超过会被 feet_contact_forces 项惩罚
        max_contact_force = 500.
        # 只保留正奖励：若为 True，总奖励因惩罚项变为负时会被截断为 0（保持训练前期探索积极性）
        only_positive_rewards = True
        # scales：奖励项权重，pass 表示完全沿用基类 LeggedRobotCfg.rewards.scales 的默认权重
        class scales( LeggedRobotCfg.rewards.scales ):
            pass

# AnymalCRoughCfgPPO：ANYmal C 崎岖地形任务的 PPO 训练配置类，继承自 LeggedRobotCfgPPO
class AnymalCRoughCfgPPO( LeggedRobotCfgPPO ):
    # runner：训练运行器（实验管理）配置
    class runner( LeggedRobotCfgPPO.runner ):
        # 本次运行名称（留空则由时间戳自动生成）
        run_name = ''
        # 实验名称：日志与模型保存目录以此命名
        experiment_name = 'rough_anymal_c'
        # 要加载的训练运行编号：-1 表示加载最近一次（最新）的运行
        load_run = -1
