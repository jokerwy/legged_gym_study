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
# 【模块说明】本文件实现 ANYmal 系列四足机器人（B/C 型号）的专用仿真环境类 Anymal。
# 它在通用基类 LeggedRobot 的基础上，增加了对"执行器网络"（actuator network，一个用
# LSTM 建模 ANYDrive 执行器动力学的循环神经网络）的支持，以更真实地模拟电机响应。
# =====================================================================

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

# 导入通用腿足机器人环境基类（本类在此基础上做扩展）
from legged_gym.envs import LeggedRobot
# 导入项目根目录常量，用于拼接执行器网络模型文件的绝对路径
from legged_gym import LEGGED_GYM_ROOT_DIR
# 导入 ANYmal C 崎岖地形配置类，作为本环境 cfg 属性的类型标注
from .mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg

# Anymal：ANYmal 机器人专用环境类，继承自 LeggedRobot。
# 核心扩展：当配置 use_actuator_network=True 时，用预训练的 LSTM 执行器网络
# 代替理想 PD 控制器计算关节力矩（模拟真实 ANYDrive 电机的滞后、带宽等动力学特性）。
class Anymal(LeggedRobot):
    # 类型标注：声明本环境的配置对象为 AnymalCRoughCfg 类型（便于 IDE 提示与静态检查）
    cfg : AnymalCRoughCfg
    # __init__：构造函数
    # 参数：cfg 环境配置；sim_params 物理仿真参数；physics_engine 物理引擎类型；
    #       sim_device 仿真计算设备（'cpu' 或 'cuda:0'）；headless 是否无渲染运行
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 调用父类构造函数：完成 Isaac Gym 场景创建、机器人加载、缓冲区初始化、奖励函数绑定等
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # load actuator network
        # 如果配置启用了执行器网络，则加载预训练的 TorchScript 执行器模型
        if self.cfg.control.use_actuator_network:
            # 将配置中的路径模板（含 {LEGGED_GYM_ROOT_DIR} 占位符）格式化为实际绝对路径
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            # 用 torch.jit.load 加载 TorchScript 序列化的 LSTM 执行器网络，并放到当前计算设备上
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
    
    # reset_idx：重置指定索引的并行环境（对父类方法的扩展重写）
    # 参数：env_ids 需要重置的环境索引张量
    def reset_idx(self, env_ids):
        # 先调用父类 reset_idx，完成机器人关节状态、基座状态、速度指令、奖励统计等常规重置
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        # 额外将被重置环境对应的执行器网络 LSTM 隐藏状态清零，避免历史信息跨 episode 残留
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        # 同理清零 LSTM 的细胞状态（cell state）
        self.sea_cell_state_per_env[:, env_ids] = 0.

    # _init_buffers：初始化各类 PyTorch 张量缓冲区（对父类方法的扩展重写）
    # 在父类缓冲区的基础上，额外创建执行器网络所需的输入张量与 LSTM 隐状态张量
    def _init_buffers(self):
        # 先调用父类方法，初始化关节状态、基座姿态、接触力、指令等常规缓冲区
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        # 执行器网络输入张量：形状 (num_envs*num_actions, 1, 2)，最后一维为 [关节位置误差, 关节速度]
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        # LSTM 隐藏状态张量：形状 (2, num_envs*num_actions, 8)，即 2 层 LSTM、隐层维度 8
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        # LSTM 细胞状态张量：形状与隐藏状态相同
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        # 将隐藏状态重塑为按环境索引的视图 (2, num_envs, num_actions, 8)，便于 reset_idx 按 env_ids 清零
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        # 同理得到细胞状态的按环境视图（与原张量共享内存）
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    # _compute_torques：根据策略输出的动作计算施加到各关节的力矩（重写父类方法）
    # 参数：actions 策略网络输出的动作张量，形状 (num_envs, num_actions)
    # 返回：关节力矩张量，形状 (num_envs, num_actions)
    # 根据配置在"执行器网络"与"理想 PD 控制器"两种方式之间二选一
    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        # 分支一：使用执行器网络计算力矩（更接近真实 ANYDrive 电机的动力学响应）
        if self.cfg.control.use_actuator_network:
            # 推理模式：前向计算不需要梯度，可加速并节省显存
            with torch.inference_mode():
                # 构造执行器网络输入第 1 个特征：关节位置误差 = 目标关节角（动作*缩放+默认角）- 当前关节角
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                # 构造执行器网络输入第 2 个特征：当前关节角速度（flatten 摊平为逐关节一维排列）
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                # 前向传播 LSTM 执行器网络：输入位置误差与速度，输出关节力矩，同时更新隐藏/细胞状态
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            # 返回执行器网络输出的力矩
            return torques
        else:
            # pd controller
            # 分支二：未启用执行器网络时，退回父类的理想 PD 控制器计算力矩
            return super()._compute_torques(actions)    