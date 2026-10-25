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
# 【模块说明】本文件定义 ANYmal B 四足机器人在崎岖地形任务下的配置。
# AnymalBRoughCfg 直接继承自 ANYmal C 的崎岖地形配置 AnymalCRoughCfg，仅覆盖与 B 型号
# 不同的模型资源字段（URDF 文件等）；AnymalBRoughCfgPPO 为对应的 PPO 训练配置。
# =====================================================================

# 从 legged_gym.envs 包入口导入 ANYmal C 崎岖地形的配置类与 PPO 配置类（作为父类复用）
from legged_gym.envs import AnymalCRoughCfg, AnymalCRoughCfgPPO

# AnymalBRoughCfg：ANYmal B 机器人崎岖地形任务的环境配置，继承自 AnymalCRoughCfg。
# ANYmal B 与 ANYmal C 的控制参数、奖励设置基本一致，因此只需替换模型资源相关字段。
class AnymalBRoughCfg( AnymalCRoughCfg ):
    # asset：模型资源配置，覆盖为 ANYmal B 的 URDF 与刚体名称
    class asset( AnymalCRoughCfg.asset ):
        # ANYmal B 机器人的 URDF 模型文件路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_b/urdf/anymal_b.urdf'
        # 机器人资产名称
        name = "anymal_b"
        # 足底刚体名称（ANYmal 系列的足底刚体统一命名为 FOOT）
        foot_name = 'FOOT'
    # rewards：奖励配置，此处不做任何修改，仅沿用父类设置
    class rewards( AnymalCRoughCfg.rewards ):
        # scales：奖励项权重，pass 表示完全沿用父类（ANYmal C 崎岖地形）的权重设置
        class scales ( AnymalCRoughCfg.rewards.scales ):
            pass

# AnymalBRoughCfgPPO：ANYmal B 任务的 PPO 训练配置，继承自 AnymalCRoughCfgPPO
class AnymalBRoughCfgPPO( AnymalCRoughCfgPPO ):
    # runner：训练运行器（实验管理）配置
    class runner ( AnymalCRoughCfgPPO.runner):
        # 本次运行名称（留空则由时间戳自动生成）
        run_name = ''
        # 实验名称：日志与模型保存目录以此命名
        experiment_name = 'rough_anymal_b'
        # 要加载的训练运行编号：-1 表示加载最近一次（最新）的运行
        load_run = -1