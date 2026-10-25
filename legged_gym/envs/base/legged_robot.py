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
# 【模块说明】腿式机器人强化学习的【核心环境基类】文件。
# 定义 LeggedRobot 类（继承 BaseTask），是整个项目最核心的环境实现：
#   1) 创建 Isaac Gym 仿真/地形/并行环境并加载机器人资产；
#   2) step() 中用 PD 控制器把策略输出的动作转成关节力矩并推进物理；
#   3) 计算观测(compute_observations)、奖励(compute_reward 及各 _reward_*)；
#   4) 实现地形课程(terrain curriculum)、指令课程(command curriculum)、
#      域随机化(摩擦/质量/推搡)等训练技巧。
# 具体机器人(如 A1/Go1)的环境类继承本类，通常只需改配置而无需改这里。
# =====================================================================

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

# 【中文】isaacgym.torch_utils 提供四元数/坐标变换等工具：
# quat_rotate_inverse(世界向量->机体系)、quat_apply(旋转向量)、to_torch 等
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
# 【中文】quat_apply_yaw: 仅用偏航角旋转向量; wrap_to_pi: 角度归一化到[-pi,pi];
# torch_rand_sqrt_float 等随机数工具
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

# 【类说明】LeggedRobot：腿式机器人环境核心类（继承 BaseTask）。
# 职责：仿真创建、动作->力矩(PD控制)、观测/奖励计算、课程学习与域随机化。
# 训练循环每次调用 step(actions) 推进一个"策略步"，内部含 decimation 个物理步。
class LeggedRobot(BaseTask):
    # 【方法说明】构造函数：解析配置 -> 创建仿真(地形+环境) -> 初始化缓冲区 -> 准备奖励函数。
    # 参数：cfg 环境配置；sim_params 仿真参数；physics_engine 物理引擎；
    #       sim_device 仿真设备；headless 是否无界面。
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        # 【中文】保存配置与仿真参数；height_samples 地形高度采样(稍后在建地形时赋值)
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        # 【中文】debug_viz: 是否绘制调试可视化; init_done: 是否完成初始化(防止初始reset误触发课程)
        self.debug_viz = False
        self.init_done = False
        # 【中文】解析配置：计算 dt、观测/奖励缩放、指令范围、最大回合步数等
        self._parse_cfg(self.cfg)
        # 【中文】调用父类构造函数：内部会调 create_sim() 建环境并 prepare_sim
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # 【中文】非无界面模式时设置相机初始位姿
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        # 【中文】初始化各种 PyTorch 缓冲区(状态张量包装、PD增益、命令、历史量等)
        self._init_buffers()
        # 【中文】根据奖励权重表准备好要调用的奖励函数列表
        self._prepare_reward_function()
        self.init_done = True

    # 【方法说明】环境推进一个"策略步"。这是 RL 训练每一步调用的核心函数。
    # 参数 actions: [num_envs, num_actions] 策略网络输出(各关节目标位置偏移)。
    # 返回: (观测, 特权观测, 奖励, 重置标志, 额外信息)。
    # 内部流程: 裁剪动作 -> 按 decimation 重复[计算力矩->施加->物理步进] -> 后处理。
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 【中文】读取动作裁剪阈值，并把动作裁剪到 [-clip, +clip] 后放到正确设备
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        # 【中文】渲染一帧（无界面模式下为空操作）
        self.render()
        # 【中文】控制降采样(decimation)循环：策略每输出一次动作，
        # 在更细的物理步长(sim dt)上重复施加力矩 decimation 次，提高控制频率。
        for _ in range(self.cfg.control.decimation):
            # 【中文】用 PD 控制器把动作转成关节力矩，形状 [num_envs, num_actions]
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # 【中文】把力矩张量交给 Isaac Gym（unwrap 取出底层 GPU 指针）
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 【中文】推进物理仿真一个 sim dt
            self.gym.simulate(self.sim)
            # 【中文】CPU 仿真时需手动取回结果(GPU 流水线则不必)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            # 【中文】刷新关节状态张量，使 dof_pos/dof_vel 反映最新物理状态
            self.gym.refresh_dof_state_tensor(self.sim)
        # 【中文】物理推进完毕后的后处理：算观测/奖励/重置等
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # 【中文】把观测裁剪到 [-clip, +clip]，防止异常值破坏训练
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # 【中文】返回标准 RL 五元组：观测、特权观测、奖励、done、info
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # 【方法说明】物理步后处理：更新基座状态 -> 回调 -> 判终止 -> 算奖励 ->
    # 重置结束环境 -> 算观测 -> 记录历史量。每个策略步调用一次。
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        # 【中文】刷新根状态(位置/姿态/速度)与接触力张量，保证读到最新物理数据
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 【中文】回合步数+1；全局步数计数器+1(用于课程/推搡的定时)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        # 【中文】以下把世界系下的速度/重力旋转变换到机体系(base frame)，
        # 作为策略观测。root_states 各段: [0:3]位置 [3:7]四元数 [7:10]线速度 [10:13]角速度。
        # base_quat: [num_envs, 4] 基座姿态四元数
        self.base_quat[:] = self.root_states[:, 3:7]
        # 【中文】base_lin_vel: [num_envs, 3] 机体系线速度
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # 【中文】base_ang_vel: [num_envs, 3] 机体系角速度
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 【中文】projected_gravity: [num_envs, 3] 重力在机体系的分量(反映机身倾斜)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 【中文】回调：指令重采样、按朝向算角速度指令、测地形高度、随机推搡
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        # 【中文】判断哪些环境需要终止(摔倒/超时)
        self.check_termination()
        # 【中文】计算本步总奖励
        self.compute_reward()
        # 【中文】取出所有需要重置的环境索引
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 【中文】重置这些环境(随机化初始状态、更新课程等)
        self.reset_idx(env_ids)
        # 【中文】计算新的观测向量
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # 【中文】记录历史量，供下一步计算动作变化率/加速度等惩罚项
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # 【中文】需要时绘制调试可视化(如高度采样点)
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # 【方法说明】检查各环境是否需要终止(回合结束)。
    # 两种终止：1) 指定部位(如基座)接触力>阈值(摔倒)；2) 超过最大回合时长(超时)。
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # 【中文】计算"终止接触部位"的接触力模长，任一部位超过 1N 即判定摔倒需重置。
        # contact_forces 形状 [num_envs, num_bodies, 3]，取 termination_contact_indices 行。
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # 【中文】超时标志：回合步数超过最大长度(超时不给终止奖励/惩罚)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # 【中文】终止 = 摔倒 或 超时（按位或合并）
        self.reset_buf |= self.time_out_buf

    # 【方法说明】重置指定环境：更新课程 -> 重置关节/根状态 -> 重采样指令 ->
    # 清零历史缓冲区 -> 记录 episode 统计信息到 extras。
    # 参数 env_ids: 需要重置的环境索引张量 [len(env_ids)]。
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # 【中文】没有需要重置的环境则直接返回
        if len(env_ids) == 0:
            return
        # update curriculum
        # 【中文】地形课程：根据本回合走过的距离决定升级/降级地形难度
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # 【中文】指令课程：因为最大指令对所有环境共享，所以只在整回合边界处更新一次
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        # 【中文】重置关节位置/速度（在默认值附近随机化）
        self._reset_dofs(env_ids)
        # 【中文】重置基座位姿/速度（放到环境原点，速度小范围随机）
        self._reset_root_states(env_ids)

        # 【中文】为重置的环境重新采样目标速度指令
        self._resample_commands(env_ids)

        # reset buffers
        # 【中文】清零这些环境的历史量与计时器，并把 reset 标志保持为 1
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        # 【中文】把本回合各奖励项的平均值(除以回合秒数归一)写入 extras["episode"] 供日志统计，
        # 然后清零累计器，开始下一回合的统计。
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        # 【中文】记录当前平均地形难度等级(课程进度)
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # 【中文】记录当前指令课程的最大 x 速度
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        # 【中文】把超时标志传给算法(用于区分超时终止，避免对超时状态做错误的自举价值估计)
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    # 【方法说明】计算本步总奖励：遍历所有非零权重的奖励函数，加权求和。
    # 每项奖励同时累加到 rew_buf(瞬时) 和 episode_sums(回合累计，用于日志)。
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # 【中文】总奖励清零
        self.rew_buf[:] = 0.
        # 【中文】逐项调用奖励函数并乘权重累加；reward_scales 已在 _prepare_reward_function 中乘过 dt
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # 【中文】可选：把负的总奖励截断为 0，避免早期惩罚过大导致机器人"宁可早终止"
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        # 【中文】终止奖励在截断之后单独加，保证它不被 only_positive_rewards 裁掉
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    # 【方法说明】拼接观测向量 obs_buf，形状 [num_envs, num_obs]。
    # 观测顺序(默认48维本体感知)：基座线速度(3)+角速度(3)+重力投影(3)+指令(3)
    #   +关节位置偏差(num_dof)+关节速度(num_dof)+上一步动作(num_actions)。
    # 若开启地形高度测量再追加高度维；可选叠加噪声模拟传感器误差。
    def compute_observations(self):
        """ Computes observations
        """
        # 【中文】按列拼接(dim=-1)各观测量，并乘各自的归一化缩放系数。
        # 关节位置用 (当前-默认) 的偏差表示；每项形状都是 [num_envs, k]。
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # 【中文】追加地形高度观测：机身高度 - 0.5 - 采样点地形高度，裁剪后缩放。
        # 这是"非盲"模式的关键，让策略感知脚下地形起伏。
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        # 【中文】叠加 [-1,1] 均匀噪声 × 各维噪声幅值，模拟真实传感器误差、提升鲁棒性
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # 【方法说明】创建仿真：实例化 Isaac Gym 仿真器，按配置创建地形，再创建所有并行环境。
    # 在父类 __init__ 中被调用。
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # 【中文】世界上方向索引：2 表示 z 轴向上(影响重力向量方向)
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        # 【中文】创建 Isaac Gym 仿真器实例
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        # 【中文】heightfield/trimesh 地形需要先用 Terrain 类生成地形几何
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        # 【中文】按地形类型分别创建：平地 / 高度场 / 三角网格
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        # 【中文】创建所有并行环境并加载机器人 actor
        self._create_envs()

    # 【方法说明】设置可视化相机的位置与注视点。
    # 参数 position/lookat: 各为 [x,y,z] 列表
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    # 【方法说明】刚体形状属性回调：在每个环境创建时被调用。
    # 默认行为：为每个环境随机化地面摩擦系数（域随机化）。
    # 参数 props: 该资产各碰撞形状的属性列表；env_id: 环境索引。返回修改后的 props。
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            # 【中文】仅在第 0 个环境时统一生成摩擦随机化表(避免重复计算)
            if env_id==0:
                # prepare friction randomization
                # 【中文】用"分桶"技巧：先生成 64 个随机摩擦值，再为每个环境随机挑一桶，
                # 这样既省内存又能让大量环境共享有限的摩擦取值。
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            # 【中文】把该环境所有碰撞形状的摩擦系数设为本环境采样到的值
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    # 【方法说明】关节(DOF)属性回调：在每个环境创建时被调用。
    # 默认行为：读取 URDF 中的关节位置/速度/力矩上限，并据此计算"软限位"。
    # 参数 props: 各关节属性数组；env_id: 环境索引。返回 props。
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        # 【中文】仅在第 0 个环境读取并缓存(所有环境共享同一资产，限位相同)
        if env_id==0:
            # 【中文】dof_pos_limits: [num_dof, 2](下/上限); dof_vel_limits/torque_limits: [num_dof]
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                # 【中文】从 URDF 读取第 i 个关节的位置下限/上限、速度上限、力矩上限
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                # 【中文】计算"软限位"：以中点为中心、按 soft_dof_pos_limit 比例缩放区间，
                # 用于奖励函数惩罚关节接近物理极限。
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    # 【方法说明】刚体属性回调：在每个环境创建时被调用。
    # 默认行为：随机化基座质量（域随机化，增强对负载变化的鲁棒性）。
    # 参数 props: 各刚体属性列表；env_id: 环境索引。返回修改后的 props。
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # 【中文】props[0] 是基座刚体，在其质量上叠加一个随机附加质量
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    # 【方法说明】物理步后回调(在算终止/奖励/观测之前调用)。
    # 默认行为：周期性重采样指令、按朝向误差算 yaw 角速度指令、测地形高度、周期性随机推搡机器人。
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 【中文】找出"到达指令重采样时刻"的环境索引(回合步数能整除重采样周期)。
        # resampling_time/dt 把秒换算成步数。
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        # 【中文】朝向指令模式：根据目标朝向与当前朝向的误差，算出 yaw 角速度指令。
        # 这样策略学的是"转向某朝向"，而不是直接给角速度。
        if self.cfg.commands.heading_command:
            # 【中文】把机体前向向量(1,0,0)用当前姿态旋转到世界系，求其朝向角
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # 【中文】yaw角速度指令 = 0.5 * 朝向误差(归一化到[-pi,pi])，再限幅到[-1,1]
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # 【中文】采样机器人周围的地形高度(供观测与基座高度奖励使用)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # 【中文】到达推搡周期时随机推一下机器人(扰动鲁棒性训练)
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    # 【方法说明】为指定环境随机重采样速度指令。
    # 参数 env_ids: 需要新指令的环境索引。指令存入 self.commands，形状 [len(env_ids), 4]。
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # 【中文】在配置范围内均匀采样 x 向线速度指令，列 0
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # 【中文】y 向线速度指令，列 1
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            # 【中文】朝向模式：采样目标朝向(列 3)
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            # 【中文】角速度模式：直接采样 yaw 角速度指令(列 2)
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        # 【中文】把 xy 速度指令模长<0.2 的置零：避免给"几乎为零但仍非零"的指令，
        # 让"站住不动"成为可学的明确目标。
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    # 【方法说明】把策略动作转换为关节力矩(核心控制律)。
    # 参数 actions: [num_envs, num_actions]。返回 torques: [num_envs, num_dof]。
    # 三种控制类型：'P'位置 / 'V'速度 / 'T'直接力矩。
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        # 【中文】动作缩放：目标关节偏移 = action * action_scale
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            # 【中文】位置 PD 控制律（最常用）：
            #   torque = kp * (q_desired - q) - kd * dq
            # 其中 q_desired = action*scale + 默认关节角(站立姿态)，q/dq 为当前关节角/角速度。
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            # 【中文】速度 PD 控制律：跟踪目标速度，微分项用速度差的数值微分
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            # 【中文】直接力矩控制：动作缩放后直接作为力矩
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # 【中文】把力矩裁剪到 URDF 力矩限位之内，保证不超出电机能力
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # 【方法说明】重置指定环境的关节状态：位置在默认角的 0.5~1.5 倍内随机，速度置零。
    # 参数 env_ids: 待重置环境索引。写回 Isaac Gym 关节状态张量。
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # 【中文】关节角 = 默认角 × [0.5,1.5] 均匀随机；形状 [len(env_ids), num_dof]
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        # 【中文】关节速度清零
        self.dof_vel[env_ids] = 0.

        # 【中文】Isaac Gym 要求索引为 int32；按索引把新关节状态写回仿真器
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    # 【方法说明】重置指定环境的基座(根)状态：位置放到环境原点(课程地形按难度放置)，
    # 速度在 [-0.5,0.5] 内随机。参数 env_ids: 待重置环境索引。
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # 【中文】课程地形(custom_origins=True)：用该环境当前难度对应的地形原点，
        # 并在 xy 方向再加 ±1m 随机偏移，增加初始位置多样性。
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            # 【中文】平地：直接放到网格状环境原点
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # 【中文】基座线速度(7:10)和角速度(10:13)在 [-0.5,0.5] 内随机初始化
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        # 【中文】按索引把新的根状态写回仿真器
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # 【方法说明】随机推搡所有机器人：通过直接设置基座 xy 速度来模拟一个瞬时冲量，
    # 用于提升策略对外力扰动的鲁棒性（域随机化的一种）。
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # 【中文】读取推搡最大速度，把基座 x/y 线速度设为 [-max_vel, max_vel] 的随机值
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # 【中文】把修改后的根状态一次性写回仿真器(对所有环境生效)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # 【方法说明】地形课程(curriculum)：根据机器人在本回合走过的距离动态调整地形难度。
    # 走得远 -> 升级更难地形；走得近 -> 降级更简单地形。参数 env_ids: 本次被重置的环境。
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        # 【中文】初始化阶段(首次 reset)不调整难度，直接返回
        if not self.init_done:
            # don't change on initial reset
            return
        # 【中文】计算本回合机器人相对其地形原点走过的水平距离
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        # 【中文】走过超过半块地形长度 -> 升级(进入更难的一行)
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        # 【中文】走的距离还不到"指令速度×回合时长"的一半 -> 降级(更简单地形)
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        # 【中文】按升级/降级更新地形难度等级
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        # 【中文】通过最难等级的机器人随机分配到某个等级(保持多样性)，否则限制在 [0, max_level)
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        # 【中文】根据新的(难度等级, 地形类型)查表更新环境原点
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    # 【方法说明】指令课程：当机器人跟踪线速度的奖励足够高时，扩大速度指令范围。
    # 参数 env_ids: 本次被重置的环境。
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # 【中文】若平均跟踪奖励超过最大值的 80%，说明当前速度范围已掌握，
        # 则把 x 速度指令范围向两侧各扩 0.5(受 max_curriculum 上限约束)。
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    # 【方法说明】构造与各观测维度对应的噪声幅值向量(noise_scale_vec)。
    # 注意：其切片索引必须与 compute_observations 中观测的拼接顺序严格一致。
    # 参数 cfg: 环境配置。返回形状 [num_obs] 的噪声幅值向量。
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # 【中文】以单个环境的观测维度为模板，创建全零噪声向量
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # 【中文】以下按观测拼接顺序为每段赋噪声幅值：噪声幅值 = 噪声scale × 总电平 × 观测scale。
        # [0:3] 基座线速度
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # 【中文】[3:6] 基座角速度
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # 【中文】[6:9] 重力投影
        noise_vec[6:9] = noise_scales.gravity * noise_level
        # 【中文】[9:12] 速度指令：不加噪声(指令是给定目标，非传感器测量)
        noise_vec[9:12] = 0. # commands
        # 【中文】[12:24] 关节位置(12个关节)
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # 【中文】[24:36] 关节速度
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # 【中文】[36:48] 上一步动作：不加噪声(动作是已知量)
        noise_vec[36:48] = 0. # previous actions
        # 【中文】[48:235] 地形高度测量(187 个采样点)
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    # 【方法说明】初始化所有 PyTorch 缓冲区：把 Isaac Gym 的底层状态张量包装成
    # PyTorch 张量(共享显存，避免拷贝)，并创建各种派生量/历史量/PD增益等缓冲区。
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        # 【中文】从 Isaac Gym 获取三类底层状态张量(原始指针)：
        # 根状态、关节状态、各刚体净接触力；随后刷新一次以拿到有效初值。
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 【中文】wrap_tensor 把底层张量包装成 PyTorch 张量(共享内存，仿真更新即自动反映)。
        # root_states: [num_envs, 13] = [位置3, 四元数4, 线速度3, 角速度3]
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # 【中文】dof_state 重塑为 [num_envs, num_dof, 2]，[...,0]为位置，[...,1]为速度
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        # 【中文】基座姿态四元数切片，[num_envs, 4]
        self.base_quat = self.root_states[:, 3:7]

        # 【中文】接触力张量重塑为 [num_envs, num_bodies, 3](xyz 三分量)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        # 【中文】全局步数计数器(跨回合累计，用于课程/推搡定时)
        self.common_step_counter = 0
        self.extras = {}
        # 【中文】构造观测噪声幅值向量
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # 【中文】世界系重力向量(0,0,-1)并复制到每个环境，[num_envs, 3]
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # 【中文】机体前向单位向量(1,0,0)并复制到每个环境，[num_envs, 3]
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        # 【中文】torques/actions/last_actions: [num_envs, num_actions]
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 【中文】p_gains/d_gains: [num_actions] 每个关节的 PD 增益(下面按名字填充)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 【中文】上一步关节速度/基座速度(用于加速度、动作平滑类惩罚)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        # 【中文】速度指令缓冲区，[num_envs, num_commands]：x速度,y速度,yaw角速度,朝向
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # 【中文】指令前 3 个分量的观测缩放系数
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        # 【中文】各足端滞空时间计时器，[num_envs, 足数](用于 feet_air_time 奖励)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        # 【中文】上一步各足端是否接触地面，[num_envs, 足数](用于接触检测滤波)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        # 【中文】机体系下的线速度/角速度/重力投影(观测用)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # 【中文】需要测高时初始化高度采样点
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        # 【中文】逐关节读取默认关节角，并按名字匹配 PD 增益。
        # stiffness/damping 配置用"键名子串"匹配关节名(如 'hip' 匹配所有含 hip 的关节)。
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            # 【中文】未匹配到增益的关节置 0 并告警(P/V 控制时)
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        # 【中文】unsqueeze(0) 把默认关节角从 [num_dof] 变为 [1, num_dof]，便于广播
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    # 【方法说明】准备奖励函数列表：扫描配置里所有非零权重项，
    # 找到对应的 self._reward_<名字> 方法，组建成列表供 compute_reward 调用。
    # 同时把权重乘以 dt（把"每秒奖励"换算成"每步奖励"），并初始化回合累计器。
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        # 【中文】遍历权重表：删除权重为 0 的项(不启用)，其余权重乘 dt 转为每步量级
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        # 【中文】为每个启用的奖励项找到同名方法 _reward_<name> 并存入列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            # 【中文】termination 奖励单独处理(在截断之后再加)，此处跳过
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            # 【中文】getattr 按名字取出方法对象，实现"配置驱动"的奖励注册
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        # 【中文】为每个奖励项创建回合累计器，[num_envs]，用于日志统计
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    # 【方法说明】向仿真添加平地(ground plane)，并按配置设置摩擦与弹性。
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        # 【中文】平面法向量设为 z 轴(水平地面)
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    # 【方法说明】向仿真添加高度场地形：用离散高度采样网格表示起伏地形。
    # 同时把高度采样存到 self.height_samples 供 _get_heights 查询。
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        # 【中文】水平/竖直方向的采样间隔与量化步长
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        # 【中文】高度场的行/列数
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        # 【中文】把地形整体平移 border_size，使坐标从边界处开始
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        # 【中文】把高度采样转成 [tot_rows, tot_cols] 的张量，用于运行时快速查询地形高度
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    # 【方法说明】向仿真添加三角网格地形(trimesh)：用三角面片精确表示复杂地形
    # (如楼梯、离散台阶)。同时保留高度采样供测高。
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        # 【中文】顶点数与三角形数
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        # 【中文】把顶点与三角形索引展平后交给 Isaac Gym 构建三角网格碰撞体
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    # 【方法说明】创建所有并行环境：
    #   1) 加载机器人 URDF/MJCF 资产并配置资产选项；
    #   2) 逐环境创建 env、调用各属性回调(摩擦/关节/质量)、创建 actor；
    #   3) 记录足端/惩罚接触/终止接触等关键刚体的索引。
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # 【中文】解析资产路径(支持 {LEGGED_GYM_ROOT_DIR} 占位符)
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # 【中文】把配置里的资产选项逐项填入 Isaac Gym 的 AssetOptions
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 【中文】加载机器人资产，并读取关节数、刚体数等元信息
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        # 【中文】读取所有刚体名与关节名；并按名字匹配出足端/惩罚接触/终止接触部位名
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        # 【中文】足端刚体名列表(用于接触力索引与滞空时间计算)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # 【中文】需要惩罚接触的部位名列表
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        # 【中文】接触即终止的部位名列表
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 【中文】把初始位置/姿态/线速度/角速度拼成 13 维初始状态向量
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 【中文】计算每个环境的原点(网格或地形课程原点)
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        # 【中文】逐环境创建：env 句柄 -> 随机出生点 -> 设置摩擦/关节/质量 -> 创建 actor
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # 【中文】出生点 = 环境原点 + xy ±1m 随机偏移
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            # 【中文】调用摩擦随机化回调并应用到资产
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            # 【中文】调用关节属性回调(读取限位)并应用
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            # 【中文】调用质量随机化回调并应用(recomputeInertia 重新计算惯性)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 【中文】记录足端刚体在 contact_forces 张量中的索引(所有环境共享同一套索引)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # 【中文】记录"需惩罚接触"部位的索引
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        # 【中文】记录"接触即终止"部位的索引
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    # 【方法说明】设置每个环境的原点(出生点参考位置)。
    # 粗糙地形(heightfield/trimesh)下由地形课程平台决定；平地则按网格均匀排列。
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            # 【中文】课程地形：custom_origins=True 表示原点由地形难度/类型决定
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            # 【中文】初始最高难度等级；不开课程时直接用最高行(全难度随机)
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            # 【中文】为每个环境随机分配初始难度等级(行)
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            # 【中文】按环境序号均匀分配地形类型(列)，保证各类型都有
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            # 【中文】地形原点查找表 [num_rows, num_cols, 3]
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            # 【中文】用 (难度等级, 地形类型) 索引出每个环境的原点
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            # 【中文】平地：custom_origins=False，按网格均匀排列所有环境
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    # 【方法说明】解析配置：计算派生量。关键有：
    #   dt(策略步长)= decimation × sim_dt；各类缩放表；最大回合步数；推搡周期(换算成步数)。
    def _parse_cfg(self, cfg):
        # 【中文】策略步长 = 控制降采样数 × 物理步长（如 4 × 0.005 = 0.02s，即 50Hz）
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        # 【中文】把奖励权重、指令范围从配置类转成普通字典，便于增删/取值
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # 【中文】非 heightfield/trimesh 地形没有难度等级概念，强制关闭地形课程
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        # 【中文】最大回合时长(秒)与对应的步数
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # 【中文】把推搡周期从秒换算成步数
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # 【方法说明】绘制调试可视化(会显著减慢仿真)。默认绘制地形高度采样点。
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        # 【中文】不测高则无需绘制
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # 【中文】用黄色线框小球标记每个高度采样点
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            # 【中文】把机体系下的采样点按偏航角旋转到世界系
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    # 【方法说明】生成高度测量的采样点(在机体系下)。
    # 把配置的 x、y 采样坐标做笛卡尔积成网格。返回 [num_envs, num_height_points, 3]。
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        # 【中文】meshgrid 生成网格：17 个 x × 11 个 y = 187 个采样点
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        # 【中文】初始化采样点张量，填入网格 xy 坐标(z 置 0，高度由查询得到)
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    # 【方法说明】采样每个机器人周围的地形高度。
    # 采样点先按基座偏航角旋转、再平移到基座位置(世界系)，然后查高度场并做双线性下取整。
    # 参数 env_ids: 可选，只对部分环境采样。返回 [num_envs, num_height_points] 的高度。
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        # 【中文】平地直接返回全零高度；'none' 地形不支持测高则报错
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # 【中文】把机体系采样点按偏航角旋转并平移到世界系坐标。
        # quat_apply_yaw 只用偏航角旋转(忽略横滚/俯仰)，因为地形高度只与水平位置有关。
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # 【中文】把世界坐标换算成高度场网格索引：先加回边界偏移，再除以水平采样间隔取整
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        # 【中文】把索引裁剪到合法范围(留出 +1 的余量供下方邻域取值)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        # 【中文】取采样点周围 3 个邻近网格的高度，取最小值(保守估计，避免穿模)。
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        # 【中文】乘竖直量化步长还原为真实高度，并重塑为 [num_envs, num_height_points]
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    # 【奖励函数总说明】每个 _reward_<name> 返回形状 [num_envs] 的张量，
    # 表示每个环境本步该项的原始奖励/惩罚值，之后由 compute_reward 乘权重累加。
    # 约定：权重为正即为"奖励"(值越大越好)，为负即为"惩罚"。

    # 【奖励】惩罚基座 z 向(竖直)线速度：抑制上下颠簸，让运动平稳。返回 [num_envs]
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    # 【奖励】惩罚基座横滚/俯仰角速度(xy 轴角速度)：保持机身姿态平稳。返回 [num_envs]
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    # 【奖励】惩罚非水平姿态：用重力在机体系 xy 分量衡量倾斜程度，越平越好。返回 [num_envs]
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # 【奖励】惩罚基座高度偏离目标高度(相对地形)：维持合适的站立高度。返回 [num_envs]
    def _reward_base_height(self):
        # Penalize base height away from target
        # 【中文】基座高度 = 基座世界高度 - 周围地形高度的均值
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    # 【奖励】惩罚关节力矩平方和：鼓励省力、降低能耗。返回 [num_envs]
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    # 【奖励】惩罚关节速度平方和：抑制过快/抖动动作。返回 [num_envs]
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    # 【奖励】惩罚关节加速度(用速度差的数值微分近似)：让动作更平滑。返回 [num_envs]
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    # 【奖励】惩罚动作变化率(相邻两步动作之差)：抑制动作抖动。返回 [num_envs]
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    # 【奖励】惩罚指定部位(非足端)发生碰撞：碰撞力>0.1N 计一次。返回 [num_envs]
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    # 【奖励】终止奖励/惩罚：回合因失败(非超时)终止时为 1(配合负权重即惩罚)。返回 [num_envs]
    def _reward_termination(self):
        # Terminal reward / penalty
        # 【中文】reset_buf 且非超时 = 真正的失败终止(摔倒)
        return self.reset_buf * ~self.time_out_buf
    
    # 【奖励】惩罚关节位置超出"软限位"：防止关节顶到物理极限。返回 [num_envs]
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        # 【中文】超出下限的量(取正)
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        # 【中文】加上超出上限的量
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # 【奖励】惩罚关节速度超出"软限位"：每关节最多计 1 rad/s 误差避免惩罚过大。返回 [num_envs]
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # 【奖励】惩罚力矩超出"软限位"：防止长期满输出损伤电机。返回 [num_envs]
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    # 【奖励】线速度跟踪奖励(核心任务奖励)：鼓励实际 xy 速度跟随指令。
    # 用高斯核 exp(-误差^2/sigma)，误差越小奖励越接近 1。返回 [num_envs]
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    # 【奖励】角速度跟踪奖励(核心任务奖励)：鼓励实际 yaw 角速度跟随指令。返回 [num_envs]
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # 【奖励】足端滞空时间奖励：奖励较长的"步幅"(抬腿时间)，促进形成清晰步态。
    # 用接触滤波抵消 PhysX 在网格地形上不稳定的接触报告。返回 [num_envs]
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # 【中文】当前足端 z 向接触力>1N 视为接触
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # 【中文】接触滤波：当前接触 或 上一步接触，任一为真则视为接触(去抖)
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        # 【中文】"首次着地"= 之前在空气中(滞空时间>0) 且 现在接触
        first_contact = (self.feet_air_time > 0.) * contact_filt
        # 【中文】滞空计时器累加
        self.feet_air_time += self.dt
        # 【中文】只在首次着地瞬间给奖励，奖励值 = 滞空时间 - 0.5s(超过0.5s的步幅才有正奖励)
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # 【中文】零指令(站着不动)时不给滞空奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # 【中文】着地后把该足滞空计时清零(接触中的足重新计 0)
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    # 【奖励】惩罚绊倒：足端水平接触力远大于竖直力(踢到竖直面/台阶侧面)。返回 [num_envs]
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        # 【中文】水平力 > 5×竖直力 即判定为踢到竖直面
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    # 【奖励】惩罚零指令时的乱动：指令为零时应保持默认站姿不动。返回 [num_envs]
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # 【中文】关节偏离默认角度的总和 × (指令为零的掩码)
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # 【奖励】惩罚足端接触力过大：超过 max_contact_force 的部分被惩罚(保护硬件)。返回 [num_envs]
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
