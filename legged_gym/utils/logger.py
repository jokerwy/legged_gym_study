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
# 【模块说明】本文件实现一个简单的日志器 Logger，主要在 play.py（策略回放）
# 阶段使用。功能包括：按步记录机器人状态（关节位置/速度/力矩、基座速度、
# 接触力等）、在独立进程中用 matplotlib 绘制 3x3 状态曲线图，
# 以及统计并打印平均回合奖励。
# =====================================================================
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

# Logger 类：回放（play）时的数据记录与可视化工具。
# state_log 记录时间序列状态；rew_log 记录各奖励项的累计值；
# 绘图放在独立子进程中执行，避免 matplotlib 阻塞仿真主循环。
class Logger:
    def __init__(self, dt):
        # 中文：初始化日志器。
        # 参数 dt - 仿真步长（秒），用于把步数换算成时间轴。
        # 状态记录字典：key -> 按时间顺序追加的数值列表（defaultdict 自动建空列表）
        self.state_log = defaultdict(list)
        # 奖励记录字典：key -> 各奖励项的累计值列表
        self.rew_log = defaultdict(list)
        self.dt = dt
        # 已统计的回合总数（用于计算平均奖励）
        self.num_episodes = 0
        # 绘图子进程句柄（None 表示尚未启动绘图）
        self.plot_process = None

    def log_state(self, key, value):
        # 中文：记录单个状态量。key - 状态名；value - 当前步的数值。
        self.state_log[key].append(value)

    def log_states(self, dict):
        # 中文：批量记录状态。dict - {状态名: 数值} 的字典，逐项追加。
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        # 中文：记录回合结束时的奖励统计。
        # 参数：dict - infos["episode"] 中的统计字典；num_episodes - 本步结束的回合数。
        # 只记录 key 中含 'rew' 的项；每项乘以回合数再累加，
        # 这样 print_rewards 时用总和除以总回合数即得平均奖励。
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        # 中文：清空所有状态与奖励记录。
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        # 中文：在独立子进程中绘制状态曲线（异步，不阻塞仿真主循环）。
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        # 中文：实际绘图函数（在子进程中运行）。生成 3x3 子图网格，
        # 分别展示：基座 x/y/yaw 速度（指令 vs 实测）、关节位置/速度（目标 vs 实测）、
        # 基座 z 向速度、足底垂直接触力、力矩-速度曲线、关节力矩。
        nb_rows = 3
        nb_cols = 3
        # 中文：创建 3 行 3 列的子图画布
        fig, axs = plt.subplots(nb_rows, nb_cols)
        # 中文：取任意一条记录的长度生成时间轴（所有记录长度相同，取到后即 break）
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        # 中文：给 state_log 起短别名，方便下面引用
        log= self.state_log
        # plot joint targets and measured positions
        # 中文：子图[1,0]：关节位置——目标位置（策略输出×action_scale）与实测位置对比
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        # 中文：子图[1,1]：关节速度——实测值与目标值对比
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        # 中文：子图[0,0]：基座 x 向线速度——指令速度 vs 实际速度（检验速度跟踪效果）
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        # 中文：子图[0,1]：基座 y 向线速度——指令 vs 实测
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        # 中文：子图[0,2]：基座 yaw 角速度——指令 vs 实测
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot base vel z
        # 中文：子图[1,2]：基座 z 向（竖直）线速度
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot contact forces
        # 中文：子图[2,0]：各足端的垂直接触力随时间变化（每条腿一条曲线）
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        # 中文：子图[2,1]：力矩-速度散点图（用于观察电机工作区间/极限）
        a = axs[2, 1]
        if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()
        # plot torques
        # 中文：子图[2,2]：关节力矩随时间变化
        a = axs[2, 2]
        if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        plt.show()

    def print_rewards(self):
        # 中文：打印各奖励项的“每秒平均奖励”及统计的总回合数。
        # 计算方式：某奖励项累计值总和 / 总回合数。
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        # 中文：析构函数——对象销毁时若绘图子进程仍在运行，则强制结束它，
        # 防止残留孤儿进程。
        if self.plot_process is not None:
            self.plot_process.kill()