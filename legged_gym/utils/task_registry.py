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
# 【模块说明】本文件实现“任务注册表”（TaskRegistry）——整个 legged_gym
# 项目的核心调度中枢。它把 任务名 -> (环境类, 环境配置, 训练配置) 的映射
# 集中管理，并提供两个关键工厂方法：
#   - make_env()：按任务名创建 Isaac Gym 仿真环境；
#   - make_alg_runner()：按任务名创建 PPO 训练 runner（rsl_rl 的 OnPolicyRunner）。
# 文件末尾创建了全局唯一实例 task_registry，各环境模块（如 legged_gym/envs/__init__.py）
# 在 import 时调用 register() 完成注册，train.py / play.py 再通过名字取用。
# =====================================================================
import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

# rsl_rl 是 ETH RSL 的强化学习库：VecEnv 是向量化环境接口基类，
# OnPolicyRunner 是 on-policy（如 PPO）算法的训练执行器
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

# 项目根目录与环境目录常量，用于拼接日志/模型保存路径
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
# 从 helpers 导入各类工具函数：参数解析、配置覆盖、类转字典、模型路径查找、随机种子、仿真参数解析
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
# 基础环境配置与 PPO 训练配置类（作为类型注解，提示注册时的配置类型）
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# 任务注册表类：维护 任务名 -> 环境类/环境配置/训练配置 三张映射表，
# 并负责环境实例与算法 runner 的创建，是 train.py / play.py 与具体任务之间的桥梁。
class TaskRegistry():
    def __init__(self):
        # 中文：初始化三张空字典，分别存放任务名到 环境类、环境配置、训练配置 的映射
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        # 中文：注册一个任务。把任务名 name 与 环境类 task_class、
        # 环境配置 env_cfg、训练配置 train_cfg 绑定存入三张字典。
        # 通常在 legged_gym/envs/__init__.py 中随包导入时自动调用。
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        # 中文：按任务名取出已注册的环境类（注意是“类”而不是实例）。
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        # 中文：按任务名取出 (环境配置, 训练配置) 二元组。
        # 同时把训练配置中的随机种子同步到环境配置，保证两边种子一致。
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        # 中文：将训练配置的种子复制给环境配置，统一随机性来源
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file

        【中文功能说明】创建 Isaac Gym 仿真环境的工厂方法，流程为：
          1) 获取命令行参数；2) 按任务名查找已注册的环境类；
          3) 加载（或使用外部传入的）环境配置；4) 用命令行参数覆盖配置；
          5) 固定随机种子；6) 解析仿真参数；7) 实例化环境类并返回。
        """
        # if no args passed get command line arguments
        # 中文：若未显式传入参数对象，则解析命令行参数（--task、--num_envs 等）
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        # 中文：检查任务名是否已注册；已注册则取出对应的环境类，
        # 否则抛出 ValueError 提示任务未注册
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            # 中文：调用方没有自带环境配置时，按任务名加载注册时保存的默认配置
            # （play.py 会先修改 env_cfg 再传入，train.py 通常走这个分支）
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        # 中文：用命令行参数覆盖配置（目前只覆盖 env_cfg.env.num_envs）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        # 中文：固定全局随机种子（numpy/torch/cuda 等），保证实验可复现
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        # 中文：把 env_cfg.sim 配置类递归转成 dict，再解析成 Isaac Gym 的 SimParams 对象
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        # 中文：实例化环境类（如 LeggedRobot）。传入：
        #   cfg            - 环境配置（含地形、奖励、观测等全部参数）
        #   sim_params     - 物理仿真参数（dt、重力、求解器等）
        #   physics_engine - 物理引擎（PhysX 或 Flex）
        #   sim_device     - 仿真计算设备（cuda:0 / cpu）
        #   headless       - 是否无界面模式（训练时通常为 True 以提速）
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        # 中文：返回创建好的环境实例与最终生效的环境配置
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file

        【中文功能说明】创建 PPO 训练算法 runner（OnPolicyRunner）的工厂方法，流程为：
          1) 获取命令行参数；2) 确定训练配置（外部传入优先，否则按任务名加载）；
          3) 用命令行参数覆盖训练配置；4) 生成带时间戳的日志目录（Tensorboard）；
          5) 实例化 OnPolicyRunner；6) 若 resume=True 则加载已有模型权重继续训练。
        """
        # if no args passed get command line arguments
        # 中文：若未显式传入参数对象，则解析命令行参数
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        # 中文：训练配置的来源优先级——外部显式传入的 train_cfg 优先；
        # 未传入时必须提供任务名 name，按名字加载注册时保存的默认训练配置；
        # 两者都没有则报错
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            # 中文：按任务名加载训练配置（同时会取出 env_cfg，但此处用不到故丢弃）
            _, train_cfg = self.get_cfgs(name)
        else:
            # 中文：若同时给了 train_cfg 和 name，则忽略 name 并打印提示
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        # 中文：用命令行参数覆盖训练配置（seed、max_iterations、resume、
        # experiment_name、run_name、load_run、checkpoint 等）
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 中文：根据 log_root 决定日志目录：
        #   "default" - 使用 <项目根>/logs/<experiment_name>/ 作为根目录；
        #   None      - 不写日志（例如纯测试时）；
        #   其他字符串 - 作为自定义日志根目录。
        if log_root=="default":
            # 中文：默认日志根目录：<项目根>/logs/<实验名>
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            # 中文：本次运行的日志目录：<日志根>/<月日_时-分-秒>_<运行名>，按时间戳区分每次运行
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # 中文：把训练配置类递归转成 dict（rsl_rl 的 OnPolicyRunner 要求 dict 形式的配置）
        train_cfg_dict = class_to_dict(train_cfg)
        # 中文：实例化 OnPolicyRunner（内部会创建 ActorCritic 网络与 PPO 算法对象）。
        # 参数：env - 向量化环境；train_cfg_dict - 算法配置；log_dir - 日志目录；
        # device - 强化学习算法所在设备（默认 cuda:0）
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        # 中文：取出是否需要断点续训的标志
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            # 中文：在日志根目录下找到指定的（或最新的）运行与 checkpoint 的完整路径
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            # 中文：把已保存的模型权重加载进 runner，实现继续训练或测试
            runner.load(resume_path)
        # 中文：返回创建好的 runner 与最终生效的训练配置
        return runner, train_cfg

# make global task registry
# 中文：创建全局唯一的任务注册表单例。各任务模块 import 本模块后通过它调用
# register() 注册任务；train.py / play.py 也通过它调用 make_env / make_alg_runner。
task_registry = TaskRegistry()