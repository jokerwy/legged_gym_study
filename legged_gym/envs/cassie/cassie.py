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
# 【模块说明】本文件实现 Cassie 双足机器人的专用仿真环境类 Cassie，继承自通用基类
# LeggedRobot。目前仅扩展了一个自定义奖励项 _reward_no_fly（奖励单脚支撑、抑制双脚腾空），
# 其余仿真流程（初始化、步进、观测与常规奖励计算等）全部复用基类实现。
# =====================================================================

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
# 导入通用腿足机器人环境基类（本类在此基础上做扩展）
from legged_gym.envs import LeggedRobot

# Cassie：Cassie 双足机器人专用环境类，继承自 LeggedRobot。
# 基类会通过命名约定自动收集所有以 _reward_ 开头的方法作为奖励项，
# 因此在这里定义 _reward_no_fly 即可为 Cassie 增加"禁止腾空"奖励（对应配置中的 no_fly 权重）。
class Cassie(LeggedRobot):
    # _reward_no_fly："禁止腾空"奖励项
    # 功能：当机器人恰好只有一只脚与地面接触（单脚支撑相）时奖励为 1，否则为 0；
    #       用于鼓励双足机器人保持行走步态（始终存在支撑脚），避免跳跃或双脚腾空。
    # 返回：形状 (num_envs,) 的张量，表示每个并行环境当前步的奖励值（0. 或 1.）
    def _reward_no_fly(self):
        # 检测每只脚的竖直方向（z 轴）接触力是否大于 0.1N，得到布尔接触标志，形状 (num_envs, 脚数)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        # 沿脚维度统计每个环境处于接触状态的脚数，判断是否恰好为 1（即单脚支撑）
        single_contact = torch.sum(1.*contacts, dim=1)==1
        # 将布尔结果转换为 0./1. 浮点奖励并返回
        return 1.*single_contact