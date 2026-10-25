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
# 【模块说明】RL 任务基类文件。定义 BaseTask：封装 Isaac Gym 仿真器
# (gym) 的获取、仿真设备选择、观测/奖励/重置等缓冲区的分配、
# 可视化窗口(viewer)与键盘事件的创建，是所有具体任务环境的抽象父类。
# LeggedRobot 继承本类并实现 create_sim / step / reset_idx 等抽象接口。
# =====================================================================

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
# 【类说明】BaseTask：强化学习任务的通用基类。
# 职责：初始化 Isaac Gym 接口、确定计算设备(CPU/GPU)、
# 按配置分配 obs/rew/reset 等并行环境缓冲区、创建仿真与渲染窗口。
# 其中 create_sim/step/reset_idx 留给子类实现（模板方法模式）。
class BaseTask():

    # 【方法说明】构造函数：完成仿真器与缓冲区的一整套初始化。
    # 参数：cfg 配置对象；sim_params 物理仿真参数；physics_engine 物理引擎类型；
    #       sim_device 仿真设备字符串(如 'cuda:0' / 'cpu')；headless 是否无界面运行。
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 【中文】获取 Isaac Gym 的全局 gym 接口句柄（后续所有 API 都通过它调用）
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        # 【中文】解析设备字符串，得到设备类型('cuda'/'cpu')与设备序号
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        # 【中文】决定环境张量所在设备：只有"仿真在GPU 且 开启GPU流水线"时才把缓冲区放GPU，
        # 否则 PhysX 会把结果拷贝回 CPU，只能用 CPU 张量。
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        # 【中文】渲染用的显卡设备号；headless(无界面)模式下置 -1 表示不渲染
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        # 【中文】从配置读取核心维度：并行环境数、观测维度、特权观测维度、动作维度
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        # 【中文】关闭 PyTorch JIT 的性能分析开关（官方示例写法，避免某些版本的开销/告警）
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # 【中文】分配核心缓冲区（张量形状与含义）：
        # obs_buf: [num_envs, num_obs] 每个环境的观测向量
        # rew_buf: [num_envs] 每个环境当前步的标量奖励
        # reset_buf: [num_envs] 重置标志(1表示需要重置)，初始全1保证第一步先复位
        # episode_length_buf: [num_envs] 当前回合已进行的步数
        # time_out_buf: [num_envs] 是否因超时结束(用于区分"超时"与"失败终止")
        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # 【中文】特权观测(如真实地形/速度等仅critic可见的信息)缓冲区，可选
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        # 【中文】额外信息字典，用于向算法层传递 episode 统计/超时等信息
        self.extras = {}

        # 【中文】创建仿真、地形与所有环境（由子类实现具体逻辑），
        # 随后 prepare_sim 完成 GPU 端内存分配与张量准备（此后才能 acquire 状态张量）
        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        # 【中文】是否让渲染与仿真同步推进；viewer 句柄先置空
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        # 【中文】非无界面模式：创建渲染窗口，并订阅 ESC(退出)、V(切换同步) 快捷键
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    # 【方法说明】返回观测缓冲区，形状 [num_envs, num_obs]
    def get_observations(self):
        return self.obs_buf
    
    # 【方法说明】返回特权观测缓冲区（可能为 None），供非对称 actor-critic 的 critic 使用
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    # 【方法说明】重置指定环境（抽象接口，由子类实现）。
    # 参数 env_ids: 需要重置的环境索引张量
    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    # 【方法说明】重置全部环境：先对所有 env 调 reset_idx，
    # 再以零动作推进一步以获得有效观测，返回 (obs, privileged_obs)
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    # 【方法说明】环境推进一步（抽象接口，由子类实现）。
    # 参数 actions: [num_envs, num_actions] 各环境的动作
    def step(self, actions):
        raise NotImplementedError

    # 【方法说明】渲染一帧：处理窗口关闭/键盘事件、按需同步图形与帧时间。
    # 参数 sync_frame_time: 是否将帧率同步到实时
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            # 【中文】若用户关闭了渲染窗口则直接退出程序
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            # 【中文】轮询键盘事件：ESC 退出；V 切换"渲染-仿真同步"开关
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            # 【中文】GPU 仿真时需先把物理结果取回，才能正确渲染最新状态
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            # 【中文】同步模式下推进图形、绘制画面并按真实时间对齐帧率；
            # 非同步模式仅轮询窗口事件（仿真可全速运行，不受渲染拖累）
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)