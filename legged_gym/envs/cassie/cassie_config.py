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
# 【模块说明】本文件定义 Cassie 双足机器人在崎岖地形（Rough Terrain）任务下的全部配置：
# CassieRoughCfg 为环境配置（观测/动作维度、地形高度采样点、初始关节角、PD 增益、奖励权重），
# CassieRoughCfgPPO 为对应的 PPO 训练配置，两者均继承自通用基类配置。
# =====================================================================

# 从基础配置模块导入通用腿足机器人环境配置基类与 PPO 训练配置基类
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# CassieRoughCfg：Cassie 双足机器人崎岖地形任务的环境配置类，继承自 LeggedRobotCfg
class CassieRoughCfg( LeggedRobotCfg ):
    # env：并行环境规模与观测/动作维度配置
    class env( LeggedRobotCfg.env):
        # 并行仿真的环境数量（GPU 大规模并行采样）
        num_envs = 4096
        # 观测向量维度：169 维（含机身状态、关节状态以及机身周围地形高度测量点等）
        num_observations = 169
        # 动作维度：Cassie 左右腿各 6 个驱动关节，共 12 个
        num_actions = 12

    
    # terrain：地形配置，此处覆盖机身周围地形高度测量点的采样网格
    class terrain( LeggedRobotCfg.terrain):
        # x 方向（机身前后）高度采样点：相对机身位置，单位米，构成 1m x 1m 网格（不含中心线）
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        # y 方向（机身左右）高度采样点：相对机身位置，单位米；11x11 网格共 121 个高度点
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    # init_state：机器人初始状态配置（基座初始位置与各关节默认角度）
    class init_state( LeggedRobotCfg.init_state ):
        # 基座初始位置 [x, y, z]，单位米；Cassie 站立时骨盆高度约 1 米
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        # 各关节默认角度（弧度），即策略输出 action = 0.0 时的目标关节角（机器人标准站姿）
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 左腿各关节默认角：髋外展、髋旋转、髋屈伸、大腿（膝）、踝、脚趾关节
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,

            # 右腿各关节默认角（与左腿镜像对称）
            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57
        }

    # control：关节控制参数（Cassie 使用理想 PD 控制器，按关节类型分别设置增益）
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # 各关节 PD 刚度（比例增益 Kp）：按键名关键词匹配关节，单位 N*m/rad
        # 髋部（外展/旋转）100、髋屈伸/大腿/踝 200、脚趾 40
        stiffness = {   'hip_abduction': 100.0, 'hip_rotation': 100.0,
                        'hip_flexion': 200., 'thigh_joint': 200., 'ankle_joint': 200.,
                        'toe_joint': 40.}  # [N*m/rad]
        # 各关节 PD 阻尼（微分增益 Kd）：单位 N*m*s/rad
        damping = { 'hip_abduction': 3.0, 'hip_rotation': 3.0,
                    'hip_flexion': 6., 'thigh_joint': 6., 'ankle_joint': 6.,
                    'toe_joint': 1.}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # 动作缩放系数：目标关节角 = 策略输出动作 * action_scale + 默认关节角
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        # 降采样系数：每个策略步（policy DT）内执行 4 次仿真+PD 控制步（sim DT）
        decimation = 4
        
    # asset：机器人模型资源配置
    class asset( LeggedRobotCfg.asset ):
        # Cassie 机器人 URDF 模型文件路径，{LEGGED_GYM_ROOT_DIR} 会被替换为项目根目录
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        # 机器人资产名称
        name = "cassie"
        # 足底刚体名称（Cassie 的"脚"为脚趾 toe，用于足端接触检测）
        foot_name = 'toe'
        # 骨盆（pelvis）接触地面时直接终止当前 episode（视为摔倒）
        terminate_after_contacts_on = ['pelvis']
        # 是否翻转视觉附件（模型外观网格）的坐标系：Cassie 模型不需要翻转
        flip_visual_attachments = False
        # 自碰撞开关：1 表示禁用自身部件间碰撞检测，0 表示启用
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    # rewards：奖励函数配置（软限制阈值、接触力上限与各奖励项权重）
    class rewards( LeggedRobotCfg.rewards ):
        # 关节位置软限制系数：关节角超过极限范围的 95% 时开始惩罚
        soft_dof_pos_limit = 0.95
        # 关节速度软限制系数：关节速度超过极限的 90% 时开始惩罚
        soft_dof_vel_limit = 0.9
        # 关节力矩软限制系数：力矩超过极限的 90% 时开始惩罚
        soft_torque_limit = 0.9
        # 足端接触力上限（牛），超过会被 feet_contact_forces 项惩罚
        max_contact_force = 300.
        # 是否只保留正奖励：False 表示允许总奖励为负（双足任务惩罚项较多，不做截断）
        only_positive_rewards = False
        # scales：各奖励项权重（正数为奖励、负数为惩罚）
        class scales( LeggedRobotCfg.rewards.scales ):
            # episode 终止（摔倒）惩罚权重：-200 重罚，强烈避免摔倒
            termination = -200.
            # 跟踪期望偏航角速度的奖励权重
            tracking_ang_vel = 1.0
            # 关节力矩惩罚权重（抑制过大输出、降低能耗）
            torques = -5.e-6
            # 关节加速度惩罚权重（使动作更平滑）
            dof_acc = -2.e-7
            # 机身竖直方向线速度惩罚权重（抑制上下颠簸）
            lin_vel_z = -0.5
            # 单脚腾空时间奖励权重（鼓励合理的步态腾空相，双足行走取较大值 5）
            feet_air_time = 5.
            # 关节位置超限惩罚权重
            dof_pos_limits = -1.
            # "禁止腾空"奖励权重：对应 Cassie 类的 _reward_no_fly，奖励恰好单脚支撑
            no_fly = 0.25
            # 关节速度惩罚权重：-0.0 表示该项实际不启用
            dof_vel = -0.0
            # 机身横滚/俯仰角速度惩罚权重：-0.0 表示不启用
            ang_vel_xy = -0.0
            # 足端接触力超限惩罚权重：-0.0 表示不启用
            feet_contact_forces = -0.

# CassieRoughCfgPPO：Cassie 崎岖地形任务的 PPO 训练配置类，继承自 LeggedRobotCfgPPO
class CassieRoughCfgPPO( LeggedRobotCfgPPO ):
    
    # runner：训练运行器（实验管理）配置
    class runner( LeggedRobotCfgPPO.runner ):
        # 本次运行名称（留空则由时间戳自动生成）
        run_name = ''
        # 实验名称：日志与模型保存目录以此命名
        experiment_name = 'rough_cassie'

    # algorithm：PPO 算法超参数
    class algorithm( LeggedRobotCfgPPO.algorithm):
        # 熵系数：策略熵正则项权重，鼓励探索
        entropy_coef = 0.01



  