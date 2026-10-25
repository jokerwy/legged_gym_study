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
# 【模块说明】本文件定义宇树科技（Unitree）A1 四足机器人在崎岖地形（Rough Terrain）任务下
# 的全部配置：A1RoughCfg 为环境配置（初始状态、PD 控制参数、模型资源、奖励权重等），
# A1RoughCfgPPO 为对应的 PPO 强化学习训练配置，两者分别继承自通用基类配置。
# =====================================================================

# 从基础配置模块导入通用腿足机器人环境配置基类与 PPO 训练配置基类
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# A1RoughCfg：宇树 A1 机器人在崎岖地形任务的环境配置类，继承自通用配置 LeggedRobotCfg。
# 通过嵌套类只覆盖需要修改的配置项，未列出的配置项全部沿用基类默认值。
class A1RoughCfg( LeggedRobotCfg ):
    # init_state：机器人初始状态配置（基座初始位置与各关节默认角度）
    class init_state( LeggedRobotCfg.init_state ):
        # 基座初始位置 [x, y, z]，单位米；0.42m 为 A1 站立时的大致机身高度
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        # 各关节默认角度（弧度），即策略输出 action = 0.0 时的目标关节角（机器人标准站姿）
        # 命名规则：FL/RL/FR/RR 分别表示 左前/左后/右前/右后 腿；hip=髋关节、thigh=大腿关节、calf=小腿关节
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 四条腿的髋关节（侧摆方向）默认角：左侧外展为正、右侧为负
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            # 四条腿的大腿关节（前后摆动方向）默认角
            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            # 四条腿的小腿关节（膝关节）默认角，负值表示小腿向后弯曲呈站姿
            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    # control：关节控制参数（控制类型、PD 增益、动作缩放、控制降采样等）
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # 控制类型：'P' 表示位置（比例）控制，即直接跟踪目标关节角；'V' 为速度控制，'T' 为力矩控制
        control_type = 'P'
        # PD 控制器刚度（比例增益 Kp）：字典键为关节名关键词，'joint' 匹配所有关节，单位 N*m/rad
        stiffness = {'joint': 20.}  # [N*m/rad]
        # PD 控制器阻尼（微分增益 Kd）：同样对所有关节统一设置，单位 N*m*s/rad
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # 动作缩放系数：目标关节角 = 策略输出动作 * action_scale + 默认关节角
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        # 降采样系数：每个策略步（policy DT）内执行 4 次仿真+PD 控制步（sim DT）
        decimation = 4

    # asset：机器人模型资源配置（URDF 文件、刚体名称、碰撞设置等）
    class asset( LeggedRobotCfg.asset ):
        # 机器人 URDF 模型文件路径，{LEGGED_GYM_ROOT_DIR} 占位符会被替换为项目根目录
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        # 机器人资产名称（Isaac Gym 中的标识符）
        name = "a1"
        # 足底刚体名称关键词，用于检测足端接触力、计算足部相关奖励
        foot_name = "foot"
        # 这些部位（大腿 thigh、小腿 calf）与地面接触时给予惩罚的刚体名关键词
        penalize_contacts_on = ["thigh", "calf"]
        # 这些部位（基座 base）接触地面时直接终止当前 episode（视为摔倒）
        terminate_after_contacts_on = ["base"]
        # 自碰撞开关：1 表示禁用机器人自身各部件间的碰撞检测，0 表示启用
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    # rewards：奖励函数配置（软限制阈值、目标量与各奖励项权重）
    class rewards( LeggedRobotCfg.rewards ):
        # 关节位置软限制系数：关节角超过其极限范围的 90% 时开始施加惩罚
        soft_dof_pos_limit = 0.9
        # 机身目标高度（米），用于 base_height 奖励项，鼓励机身保持在该高度附近
        base_height_target = 0.25
        # scales：各奖励项权重（正数为奖励、负数为惩罚），未列出项沿用基类默认权重
        class scales( LeggedRobotCfg.rewards.scales ):
            # 关节力矩惩罚权重：抑制过大的力矩输出，降低能耗与机械磨损
            torques = -0.0002
            # 关节位置超限惩罚权重：强烈惩罚关节角超出软限制范围
            dof_pos_limits = -10.0

# A1RoughCfgPPO：A1 崎岖地形任务的 PPO 训练配置类，继承自 LeggedRobotCfgPPO
class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    # algorithm：PPO 算法超参数
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # 熵系数：策略熵正则项权重，越大越鼓励探索
        entropy_coef = 0.01
    # runner：训练运行器（实验管理）配置
    class runner( LeggedRobotCfgPPO.runner ):
        # 本次运行名称（留空则由时间戳自动生成）
        run_name = ''
        # 实验名称：日志与模型权重的保存目录以此命名
        experiment_name = 'rough_a1'

  