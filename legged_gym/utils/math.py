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
# 【模块说明】本文件提供 legged_gym 中常用的数学工具函数（基于 PyTorch）。
# 包括：仅按偏航角旋转三维向量、把角度归一化到 [-pi, pi]、
# 按“平方根分布”采样随机浮点数。这些函数主要被环境类
# （legged_robot.py 等）在计算观测、奖励和指令重采样时调用。
# =====================================================================
import torch
from torch import Tensor
import numpy as np
# 从 Isaac Gym 自带的 torch 工具库导入：四元数旋转向量函数、向量归一化函数
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    # 仅用四元数的偏航（yaw，绕竖直 z 轴）分量旋转向量，忽略滚转/俯仰分量。
    # 参数：quat - 形状 (N, 4) 的四元数（x, y, z, w）；vec - 形状 (N, 3) 的向量。
    # 返回：仅按偏航角旋转后的向量。常用于把机器人本体系向量投影到“水平航向系”。
    # 克隆并展平为 (N, 4)，避免修改原张量
    quat_yaw = quat.clone().view(-1, 4)
    # 将四元数的 x、y 分量置零，只保留 z、w 分量，即只保留绕竖直轴的旋转
    quat_yaw[:, :2] = 0.
    # 归一化得到合法的纯偏航四元数
    quat_yaw = normalize(quat_yaw)
    # 用纯偏航四元数旋转向量并返回
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    # 把任意角度（弧度）归一化到 (-pi, pi] 区间。
    # 参数：angles - 角度张量/数组；返回：归一化后的角度。
    # 先对 2π 取模，把角度映射到 [0, 2π)
    angles %= 2*np.pi
    # 大于 π 的角度减去 2π，映射到负区间，最终结果落在 (-π, π]
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # 在 [lower, upper] 区间按“平方根变换后的均匀分布”采样随机数。
    # 这种分布在两端（靠近 lower 和 upper）的概率密度高于中间，
    # 常用于电机零漂（motor zero position）等需要偏向极值的随机化。
    # 参数：lower/upper - 采样上下界；shape - 输出张量形状；device - 计算设备。
    # 返回：形状为 shape 的随机张量。
    # type: (float, float, Tuple[int, int], str) -> Tensor
    # 先在 [-1, 1] 上均匀采样
    r = 2*torch.rand(*shape, device=device) - 1
    # 对符号保留地开平方：负值取 -sqrt(-r)，正值取 sqrt(r)，使样本向 ±1 两端聚集
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    # 把 [-1, 1] 映射回 [0, 1]
    r =  (r + 1.) / 2.
    # 再线性映射到目标区间 [lower, upper]
    return (upper - lower) * r + lower