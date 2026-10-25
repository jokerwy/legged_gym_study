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
# 【模块说明】本文件是 legged_gym.utils 工具包的入口（初始化文件）。
# 它把 utils 子模块中最常用的函数/类集中导出，使外部代码只需
# `from legged_gym.utils import xxx` 即可使用全部工具，
# 例如 train.py / play.py 中的 get_args、task_registry、Logger 等。
# =====================================================================
# 从 helpers 导入：配置类转字典、查找模型加载路径、解析命令行参数、
# 导出 JIT 策略、设置随机种子、用字典更新配置类 等工具函数
from .helpers import class_to_dict, get_load_path, get_args, export_policy_as_jit, set_seed, update_class_from_dict
# 从 task_registry 导入：全局唯一的任务注册表单例（负责注册任务、创建环境与算法 runner）
from .task_registry import task_registry
# 从 logger 导入：回放阶段用于记录并绘制状态曲线、统计奖励的日志器
from .logger import Logger
# 从 math 导入全部数学工具函数（四元数偏航旋转、角度归一化、随机数采样等）
from .math import *
# 从 terrain 导入：地形生成类（高度场地形/三角网格地形）
from .terrain import Terrain