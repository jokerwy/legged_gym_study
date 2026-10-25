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
# 【模块说明】本文件是 legged_gym 的【策略回放/测试脚本】（play）。
# 功能：加载训练好的策略 checkpoint，在仿真环境中运行并可视化，同时可以：
#   - 把策略导出为 TorchScript（JIT）模型，供 C++ 部署；
#   - 记录并绘制机器人状态曲线（关节位置/速度/力矩、基座速度跟踪等）；
#   - 统计并打印各奖励项的平均值；
#   - 可选地录制画面帧、移动相机。
# 典型用法：python play.py --task=anymal_c_flat
# =====================================================================
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

# 中文：必须先 import isaacgym，再 import 其他依赖它的模块
import isaacgym
# 中文：导入环境包（触发任务注册）与工具函数：参数解析、JIT 导出、任务注册表、日志器
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    # 中文：回放主函数。参数 args 为命令行参数（关键：--task 指定要回放的任务，
    # --load_run / --checkpoint 指定加载哪次训练的哪个模型）。
    # 中文：按任务名取出已注册的环境配置与训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # 中文：为“测试/回放”覆盖一批配置参数，使场景更干净、确定：
    # 中文：并行环境数限制到最多 50 个（回放不需要训练时那么多环境，节省资源）
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    # 中文：地形只保留 5 行 x 5 列共 25 个子地形（便于观察）
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    # 中文：关闭课程学习（子地形难度不再按行递进）
    env_cfg.terrain.curriculum = False
    # 中文：关闭观测噪声（测试时给策略“干净”的观测）
    env_cfg.noise.add_noise = False
    # 中文：关闭摩擦系数随机化（域随机化的一种）
    env_cfg.domain_rand.randomize_friction = False
    # 中文：关闭随机推搡机器人（另一种域随机化扰动）
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    # 中文：用修改后的 env_cfg 创建仿真环境（传入 env_cfg 使 make_env 跳过默认配置加载）
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 中文：获取初始观测张量（供循环第一步输入策略）
    obs = env.get_observations()
    # load policy
    # 中文：打开 resume 标志，让 make_alg_runner 在创建 runner 时加载已训练的模型权重
    train_cfg.runner.resume = True
    # 中文：创建 runner 并加载 checkpoint（传入 train_cfg 使外部修改生效）
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # 中文：取出“推理模式”的策略函数（内部关闭 dropout/归一化更新、包装成
    # obs -> action 的可调用对象），device 与环境一致
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    # 中文：可选——把策略导出为 TorchScript JIT 模型，供无 Python 环境（如 C++）部署
    if EXPORT_POLICY:
        # 中文：导出路径：<项目根>/logs/<experiment_name>/exported/policies
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        # 中文：对 actor_critic 执行 JIT 导出（MLP 策略 -> policy_1.pt；
        # LSTM 策略 -> policy_lstm_1.pt）
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # 中文：创建日志器，dt 为仿真步长（用于绘图的时间轴）
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    # 中文：robot_index：记录第 0 号机器人的数据
    joint_index = 1 # which joint is used for logging
    # 中文：joint_index：记录该机器人第 1 号关节的数据
    stop_state_log = 100 # number of steps before plotting states
    # 中文：stop_state_log：记录前 100 步的状态，第 100 步时弹出绘图
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    # 中文：stop_rew_log：统计约一个回合长度的奖励，到点打印平均奖励
    # 中文：相机初始位置取自配置 viewer.pos
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # 中文：相机平移速度（MOVE_CAMERA 开启时相机以此速度移动）
    camera_vel = np.array([1., 1., 0.])
    # 中文：相机朝向 = 观察点 lookat - 相机位置
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    # 中文：录制帧的图片序号
    img_idx = 0

    # 中文：回放主循环——共运行 10 个回合长度的步数
    for i in range(10*int(env.max_episode_length)):
        # 中文：用当前观测前向推理得到动作（detach 断开梯度图，纯推理）
        actions = policy(obs.detach())
        # 中文：环境推进一步，返回：新观测、特权观测（此处忽略）、奖励、
        # 回合结束标志、附加信息（含 episode 统计）
        obs, _, rews, dones, infos = env.step(actions.detach())
        # 中文：可选——录制帧：每隔一步把渲染画面写成 png 文件
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        # 中文：可选——相机随时间平移（巡游视角）
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # 中文：前 stop_state_log 步内持续记录状态数据
        if i < stop_state_log:
            logger.log_states(
                {
                    # 中文：关节目标位置 = 策略输出动作 x 动作缩放系数
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    # 中文：关节实际位置 / 实际速度 / 实际力矩
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    # 中文：速度指令（x 向、y 向、yaw 转向）
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    # 中文：基座实际速度（x/y/z 线速度、yaw 角速度）
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    # 中文：各足端的垂直接触力（feet_indices 为足端刚体索引，取 z 分量）
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            # 中文：到达第 stop_state_log 步——在子进程中绘制全部状态曲线
            logger.plot_states()
        # 中文：统计区间 (0, stop_rew_log) 内：若有环境结束了回合
        # （infos["episode"] 非空），累计奖励统计
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                # 中文：本步结束的回合数 = reset 标志为真的环境数量
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            # 中文：到达第 stop_rew_log 步——打印各奖励项的平均值
            logger.print_rewards()

if __name__ == '__main__':
    # 中文：回放功能开关（按需修改后运行脚本）：
    # EXPORT_POLICY - 是否导出 JIT 策略模型（默认开）
    EXPORT_POLICY = True
    # RECORD_FRAMES - 是否逐帧保存渲染画面（默认关）
    RECORD_FRAMES = False
    # MOVE_CAMERA - 是否让相机随时间移动（默认关）
    MOVE_CAMERA = False
    # 中文：解析命令行参数并执行回放
    args = get_args()
    play(args)
