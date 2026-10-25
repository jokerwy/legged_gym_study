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
# 【模块说明】本文件是 legged_gym 的通用工具函数集合，主要包括：
#   - class_to_dict / update_class_from_dict：配置类与字典互转（嵌套递归）；
#   - set_seed：统一固定 python/numpy/torch/cuda 的随机种子；
#   - parse_sim_params：把配置与命令行参数解析成 Isaac Gym 的 SimParams；
#   - get_load_path：在日志目录中定位要加载的模型 checkpoint 文件；
#   - update_cfg_from_args：用命令行参数覆盖环境/训练配置；
#   - get_args：定义并解析全部命令行参数；
#   - export_policy_as_jit：把训练好的策略导出为 TorchScript（JIT）模型，
#     便于在 C++ 等无 Python 环境中部署；PolicyExporterLSTM 是循环策略的导出器。
# =====================================================================
import os
import copy
import torch
import numpy as np
import random
# Isaac Gym 的 API 定义（gymapi）与实用工具（gymutil）
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    # 中文：把“配置类对象”递归转换成嵌套字典（rsl_rl 的 OnPolicyRunner
    # 等接口要求 dict 形式的配置）。
    # 参数：obj - 任意对象；返回：若 obj 是普通值则原样返回，
    # 若是类实例则返回 {属性名: 递归转换后的值} 的字典。
    # 中文：没有 __dict__ 的对象（int/float/str 等基本类型）直接返回原值
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    # 中文：遍历对象的所有属性名
    for key in dir(obj):
        # 中文：跳过下划线开头的私有/魔术属性
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        # 中文：列表类型逐项递归转换；其他类型直接递归转换
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    # 中文：class_to_dict 的逆操作——用字典中的键值递归更新配置类对象的属性。
    # 若属性本身是嵌套的“类”，则递归进入更新；否则直接 setattr 赋值。
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        # 中文：attr 是类（type）说明这是嵌套配置节，递归更新其成员
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    # 中文：统一固定所有随机源的种子，保证实验可复现。
    # 参数：seed - 随机种子；若为 -1 则随机生成一个 [0, 10000) 的种子。
    # 返回：无（直接修改各库的全局种子）。
    # 中文：-1 表示“随机选种子”
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    # 中文：分别固定 python 内置 random、numpy、torch（CPU）、
    # PYTHONHASHSEED 环境变量（影响 hash 随机化）、torch CUDA 的单卡与多卡种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # 中文：根据命令行参数 args 和配置字典 cfg 构建 Isaac Gym 的仿真参数对象
    # SimParams（含物理引擎选择、GPU 管线、dt、重力、求解器等）。
    # 返回：gymapi.SimParams 对象。
    # code from Isaac Gym Preview 2
    # initialize sim params
    # 中文：创建默认的 SimParams 对象
    sim_params = gymapi.SimParams()

    # set some values from args
    # 中文：根据所选物理引擎做相应设置——
    # Flex 引擎只支持 CPU（在非 cpu 设备上打印警告）；
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # 中文：PhysX 引擎：设置是否使用 GPU 以及子场景数量
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    # 中文：是否启用 GPU 加速管线（张量 API 需要）
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    # 中文：若配置字典里有 "sim" 节（来自 env_cfg.sim），则用其中的值
    # 解析并覆盖上面的默认设置（如 dt、重力、接触参数等）
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    # 中文：命令行显式指定了线程数时，覆盖 PhysX 的 CPU 线程数
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    # 中文：在日志根目录 root 下定位要加载的模型文件完整路径。
    # 参数：root - 日志根目录（.../logs/<experiment_name>）；
    #       load_run - 指定运行目录名，-1 表示取最新一次运行；
    #       checkpoint - 指定 checkpoint 编号，-1 表示取最新的模型。
    # 返回：模型文件（model_xxx.pt）的完整路径。
    try:
        # 中文：列出日志根目录下所有运行目录并按名称排序
        # （目录名以时间戳开头，故字典序≈时间序）
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        # 中文：排除 exported 子目录（存放导出策略，不是训练运行）
        if 'exported' in runs: runs.remove('exported')
        # 中文：最新一次运行 = 排序后的最后一个
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    # 中文：-1 表示加载最新运行；否则拼接指定的运行目录名
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    # 中文：-1 表示加载该运行中最新的 checkpoint：
    # 筛出所有含 'model' 的文件，按补零后的名字排序取最后一个
    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        # 中文：否则按指定编号拼出文件名 model_<编号>.pt
        model = "model_{}.pt".format(checkpoint) 

    # 中文：拼接得到模型文件完整路径
    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # 中文：用命令行参数覆盖配置（命令行优先级高于配置文件）。
    # 参数：env_cfg - 环境配置（可为 None）；cfg_train - 训练配置（可为 None）；
    # args - 解析后的命令行参数。返回：更新后的 (env_cfg, cfg_train)。
    # 规则：命令行参数为 None 表示用户没指定，保留配置文件中的值。
    # seed
    if env_cfg is not None:
        # num envs
        # 中文：覆盖并行环境数量（--num_envs）
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        # 中文：覆盖随机种子（--seed）
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        # 中文：以下覆盖 runner 相关参数：最大迭代数、是否断点续训、
        # 实验名、运行名、加载的运行目录、checkpoint 编号
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    # 中文：定义并解析全部命令行参数（在 Isaac Gym 自带参数基础上追加本项目自定义参数）。
    # 返回：argparse.Namespace 对象，包含 --task、--num_envs、--seed 等所有参数。
    # 中文：自定义参数列表，每个元素描述一个参数的名字/类型/默认值/帮助信息：
    custom_parameters = [
        # --task：任务名（默认 anymal_c_flat），决定使用哪个已注册的环境
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        # --resume：是否从 checkpoint 继续训练（开关标志，出现即为 True）
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        # --experiment_name：实验名，决定日志保存的子目录
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        # --run_name：本次运行名，会附加在时间戳后作为日志目录名
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        # --load_run：断点续训时加载哪个运行目录，-1 表示最新一次
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        # --checkpoint：加载哪个编号的模型文件，-1 表示最新的
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        # --headless：无界面模式（不弹出可视化窗口，训练时显著提速）
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        # --horovod：是否使用 horovod 做多 GPU 分布式训练
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        # --rl_device：强化学习算法运行的设备（默认 cuda:0）
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        # --num_envs：并行环境数量（覆盖配置文件）
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        # --seed：随机种子（覆盖配置文件）
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        # --max_iterations：最大训练迭代数（覆盖配置文件）
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    # 中文：调用 Isaac Gym 的参数解析器（它自带 --sim_device、--pipeline、
    # --graphics_device_id 等参数），并合并上面的自定义参数一起解析
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    # 中文：参数名对齐——Isaac Gym 用 compute_device_id / sim_device_type 命名，
    # 本项目统一使用 sim_device_id / sim_device
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    # 中文：若仿真设备是 cuda，则补上设备序号，形成 'cuda:0' 这样的完整设备字符串
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    # 中文：把训练好的策略网络导出为 TorchScript（JIT）文件，供 C++ 部署使用。
    # 参数：actor_critic - rsl_rl 的 ActorCritic 模型；path - 导出目录。
    # 中文：若模型带有 memory_a 属性，说明是循环（LSTM）策略，走专用导出器
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        # 中文：普通前馈（MLP）策略：直接对 actor 网络做 torch.jit.script 序列化
        os.makedirs(path, exist_ok=True)
        # 中文：导出文件名固定为 policy_1.pt
        path = os.path.join(path, 'policy_1.pt')
        # 中文：深拷贝 actor 网络并移到 CPU（导出与部署无关的 GPU 状态）
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        # 中文：用 TorchScript 脚本化模型并保存
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


# PolicyExporterLSTM 类：LSTM（循环）策略的 JIT 导出器。
# 把 actor 网络与 LSTM 记忆单元包装成一个 nn.Module，内部维护隐藏/细胞状态，
# 使整个循环策略可以作为一个整体被 torch.jit.script 导出并在 C++ 中运行。
class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        # 中文：参数：actor_critic - rsl_rl 的循环 ActorCritic 模型。
        super().__init__()
        # 中文：深拷贝 actor 输出头（MLP 部分）
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        # 中文：深拷贝 LSTM 记忆单元并移到 CPU
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        # 中文：注册隐藏状态与细胞状态为 buffer（随模型保存/移动，但不参与梯度）
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        # 中文：前向传播：输入观测 x -> LSTM -> actor -> 输出动作。
        # unsqueeze(0) 增加时间步维度（LSTM 要求 [seq_len, batch, feature]），
        # 同时把上一步的隐藏/细胞状态喂入并更新为新状态（原地赋值）。
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        # 中文：重置 LSTM 的隐藏/细胞状态（每回合开始时调用）。
        # @torch.jit.export 使该方法在导出的 TorchScript 模型中也可被 C++ 调用。
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        # 中文：把包装好的循环策略整体脚本化并保存为 policy_lstm_1.pt。
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
