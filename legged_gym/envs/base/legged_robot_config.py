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
# 【模块说明】腿式机器人的默认配置文件。定义两个配置类：
#   LeggedRobotCfg    —— 环境配置（环境数/地形/指令/初始状态/控制/资产/
#                        域随机化/奖励权重/归一化/观测噪声/相机/物理仿真参数）。
#   LeggedRobotCfgPPO —— PPO 算法配置（网络结构/训练超参/运行与日志）。
# 二者都继承 BaseConfig，通过嵌套 class 声明、属性链访问（如 cfg.env.num_envs）。
# 具体机器人（如 A1、Go1）的配置类继承本类并覆盖相应字段。
# =====================================================================

from .base_config import BaseConfig

# 【类说明】LeggedRobotCfg：腿式机器人环境的默认配置。
# 每个嵌套 class 对应一个配置分组，字段即配置项。
class LeggedRobotCfg(BaseConfig):
    # 【分组】env：并行环境/观测/动作等核心维度
    class env:
        # 【中文】并行仿真的环境数量（GPU 并行采样规模）
        num_envs = 4096
        # 【中文】观测向量维度：48 维本体感知 + 187 维地形高度 = 235
        num_observations = 235
        # 【中文】特权观测维度；None 表示不使用（非对称 actor-critic 时设为非 None）
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        # 【中文】动作维度 = 关节数（如 12 个关节的四足机器人）
        num_actions = 12
        # 【中文】平面地形时相邻环境的间距[m]（heightfield/trimesh 地形不使用）
        env_spacing = 3.  # not used with heightfields/trimeshes 
        # 【中文】是否把"超时"信息传给算法（区分超时终止与失败终止）
        send_timeouts = True # send time out information to the algorithm
        # 【中文】单个回合的时长[s]
        episode_length_s = 20 # episode length in seconds

    # 【分组】terrain：地形生成与地形课程
    class terrain:
        # 【中文】地形网格类型：none/plane(平地)/heightfield(高度场)/trimesh(三角网格)
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # 【中文】高度场水平方向采样间隔[m]
        horizontal_scale = 0.1 # [m]
        # 【中文】高度场竖直方向量化步长[m]（高度采样值乘以此值得到真实高度）
        vertical_scale = 0.005 # [m]
        # 【中文】地形四周额外留出的边界宽度[m]（防止机器人走出地形）
        border_size = 25 # [m]
        # 【中文】是否启用"地形课程"：走得远升级更难地形，走得近降级
        curriculum = True
        # 【中文】地面静摩擦系数
        static_friction = 1.0
        # 【中文】地面动摩擦系数
        dynamic_friction = 1.0
        # 【中文】地面恢复系数(弹性)，0 表示非弹性碰撞
        restitution = 0.
        # rough terrain only:
        # 【中文】是否在机器人周围采样地形高度（为策略提供本体周围的高度感知）
        measure_heights = True
        # 【中文】高度采样点的 x 坐标集合[m]：机身前方1.6m 范围
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        # 【中文】高度采样点的 y 坐标集合[m]：机身左右1.0m 范围
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # 【中文】是否只选用某一种地形类型（配合 terrain_kwargs 使用）
        selected = False # select a unique terrain type and pass all arguments
        # 【中文】selected=True 时传给该地形类型的参数字典
        terrain_kwargs = None # Dict of arguments for selected terrain
        # 【中文】初始时的最高地形难度等级（课程起点）
        max_init_terrain_level = 5 # starting curriculum state
        # 【中文】单块地形的长度[m]
        terrain_length = 8.
        # 【中文】单块地形的宽度[m]
        terrain_width = 8.
        # 【中文】地形行数 = 难度等级数（沿课程递进方向）
        num_rows= 10 # number of terrain rows (levels)
        # 【中文】地形列数 = 不同地形类型/变体数量
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # 【中文】各地形类型所占比例：[平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散台阶]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        # 【中文】坡度阈值：超过该坡度的斜坡在 trimesh 中被修正为竖直面
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    # 【分组】commands：速度指令（期望机器人跟踪的目标速度）及其课程
    class commands:
        # 【中文】是否启用"指令课程"：跟踪奖励足够高时扩大指令范围
        curriculum = False
        # 【中文】指令课程中线速度范围的上限[m/s]
        max_curriculum = 1.
        # 【中文】指令维度：默认4 = [x线速度, y线速度, yaw角速度, 朝向heading]
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # 【中文】指令重采样的时间间隔[s]（每隔这么久换一个目标速度）
        resampling_time = 10. # time before command are changed[s]
        # 【中文】是否用"朝向指令"模式：True 时由目标朝向误差算出 yaw 角速度指令
        heading_command = True # if true: compute ang vel command from heading error
        # 【子分组】ranges：各指令分量的采样范围 [最小值, 最大值]
        class ranges:
            # 【中文】x 方向线速度指令范围[m/s]
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # 【中文】y 方向线速度指令范围[m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # 【中文】yaw 角速度指令范围[rad/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            # 【中文】目标朝向范围[rad]（heading_command=True 时使用）
            heading = [-3.14, 3.14]

    # 【分组】init_state：机器人初始状态
    class init_state:
        # 【中文】基座初始位置 x,y,z [m]
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        # 【中文】基座初始姿态四元数 x,y,z,w
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        # 【中文】基座初始线速度 x,y,z [m/s]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        # 【中文】基座初始角速度 x,y,z [rad/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # 【中文】各关节的默认角度[rad]：动作=0 时 PD 控制的目标角（站立姿态）
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    # 【分组】control：底层控制器（PD 控制）参数
    class control:
        # 【中文】控制类型：'P'位置控制 / 'V'速度控制 / 'T'直接力矩控制
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        # 【中文】PD 比例增益 kp（刚度）[N*m/rad]，按键名子串匹配到具体关节
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        # 【中文】PD 微分增益 kd（阻尼）[N*m*s/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # 【中文】动作缩放系数：目标角 = action_scale * action + 默认角
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # 【中文】控制降采样：每策略步执行 decimation 次物理步（力矩按 sim dt 重复施加）
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    # 【分组】asset：机器人资产(URDF/MJCF)加载选项
    class asset:
        # 【中文】机器人模型文件路径（可含 {LEGGED_GYM_ROOT_DIR} 占位符）
        file = ""
        # 【中文】actor 名称（用于在环境中索引该机器人）
        name = "legged_robot"  # actor name
        # 【中文】足端刚体名（用于从接触力/状态张量中索引足底）
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        # 【中文】这些部位发生碰撞会被惩罚（如大腿/小腿）
        penalize_contacts_on = []
        # 【中文】这些部位发生碰撞会判定回合终止（如基座触地）
        terminate_after_contacts_on = []
        # 【中文】是否禁用重力
        disable_gravity = False
        # 【中文】是否合并固定关节连接的刚体（减少自由度、加速仿真）
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        # 【中文】是否固定基座（调试用）
        fix_base_link = False # fixe the base of the robot
        # 【中文】DOF 驱动模式：0无/1位置目标/2速度目标/3力矩(effort)
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # 【中文】自碰撞开关：1 禁用, 0 启用（按位过滤）
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        # 【中文】把碰撞体圆柱替换为胶囊体（更快更稳定）
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        # 【中文】翻转视觉网格坐标系（部分 .obj 需从 y-up 翻转为 z-up）
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        # 【中文】密度（用于由几何体积估算质量）
        density = 0.001
        # 【中文】角阻尼
        angular_damping = 0.
        # 【中文】线阻尼
        linear_damping = 0.
        # 【中文】最大角速度上限[rad/s]（数值稳定用）
        max_angular_velocity = 1000.
        # 【中文】最大线速度上限[m/s]
        max_linear_velocity = 1000.
        # 【中文】电枢(armature)：关节处附加的等效惯量，提升数值稳定性
        armature = 0.
        # 【中文】碰撞体厚度（影响接触判定裕量）
        thickness = 0.01

    # 【分组】domain_rand：域随机化（缩小 sim-to-real 差距）
    class domain_rand:
        # 【中文】是否随机化地面摩擦系数
        randomize_friction = True
        # 【中文】摩擦系数随机范围
        friction_range = [0.5, 1.25]
        # 【中文】是否随机化基座附加质量
        randomize_base_mass = False
        # 【中文】基座附加质量随机范围[kg]
        added_mass_range = [-1., 1.]
        # 【中文】是否周期性随机推搡机器人（扰动鲁棒性训练）
        push_robots = True
        # 【中文】推搡时间间隔[s]
        push_interval_s = 15
        # 【中文】推搡时给基座施加的 xy 最大速度[m/s]
        max_push_vel_xy = 1.

    # 【分组】rewards：奖励函数各项权重及相关参数
    class rewards:
        # 【子分组】scales：各奖励项的权重（正值为奖励，负值为惩罚）
        # 键名对应 LeggedRobot 中的 _reward_<键名> 方法；0 表示不启用该项。
        class scales:
            # 【中文】终止惩罚（回合因失败终止时给出）
            termination = -0.0
            # 【中文】线速度跟踪奖励（鼓励跟随目标 xy 速度）
            tracking_lin_vel = 1.0
            # 【中文】角速度跟踪奖励（鼓励跟随目标 yaw 角速度）
            tracking_ang_vel = 0.5
            # 【中文】基座 z 向线速度惩罚（抑制上下跳动）
            lin_vel_z = -2.0
            # 【中文】基座横滚/俯仰角速度惩罚（保持机身平稳）
            ang_vel_xy = -0.05
            # 【中文】姿态惩罚（抑制机身倾斜）
            orientation = -0.
            # 【中文】力矩惩罚（鼓励省力、减小能耗）
            torques = -0.00001
            # 【中文】关节速度惩罚
            dof_vel = -0.
            # 【中文】关节加速度惩罚（让动作更平滑）
            dof_acc = -2.5e-7
            # 【中文】基座高度惩罚（偏离目标高度时）
            base_height = -0. 
            # 【中文】足端滞空时间奖励（鼓励充分抬腿、形成步态）
            feet_air_time =  1.0
            # 【中文】非足端部位碰撞惩罚
            collision = -1.
            # 【中文】足部踢到竖直面(绊倒)惩罚
            feet_stumble = -0.0 
            # 【中文】动作变化率惩罚（抑制动作抖动）
            action_rate = -0.01
            # 【中文】零指令时的乱动惩罚（站立静止）
            stand_still = -0.

        # 【中文】是否把负的总奖励截断为 0（避免因早期惩罚过大导致机器人"宁可早死"）
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 【中文】速度跟踪奖励的高斯核宽度：reward = exp(-误差^2 / sigma)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # 【中文】软关节位置限位（占 URDF 限位的比例，超过则被惩罚）
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        # 【中文】软关节速度限位比例
        soft_dof_vel_limit = 1.
        # 【中文】软力矩限位比例
        soft_torque_limit = 1.
        # 【中文】基座目标高度[m]
        base_height_target = 1.
        # 【中文】足端接触力上限[N]，超过则被惩罚
        max_contact_force = 100. # forces above this value are penalized

    # 【分组】normalization：观测/动作的归一化与截断
    class normalization:
        # 【子分组】obs_scales：把各观测量缩放到相近量级，利于网络训练
        class obs_scales:
            # 【中文】线速度观测缩放
            lin_vel = 2.0
            # 【中文】角速度观测缩放
            ang_vel = 0.25
            # 【中文】关节位置观测缩放
            dof_pos = 1.0
            # 【中文】关节速度观测缩放
            dof_vel = 0.05
            # 【中文】地形高度观测缩放
            height_measurements = 5.0
        # 【中文】观测截断范围 [-clip, +clip]
        clip_observations = 100.
        # 【中文】动作截断范围 [-clip, +clip]
        clip_actions = 100.

    # 【分组】noise：向观测添加噪声（模拟传感器误差、提升鲁棒性）
    class noise:
        # 【中文】是否给观测加噪声
        add_noise = True
        # 【中文】噪声总电平（整体缩放各项噪声）
        noise_level = 1.0 # scales other values
        # 【子分组】noise_scales：各观测量的噪声幅值
        class noise_scales:
            # 【中文】关节位置噪声幅值
            dof_pos = 0.01
            # 【中文】关节速度噪声幅值
            dof_vel = 1.5
            # 【中文】线速度噪声幅值
            lin_vel = 0.1
            # 【中文】角速度噪声幅值
            ang_vel = 0.2
            # 【中文】重力投影噪声幅值
            gravity = 0.05
            # 【中文】地形高度测量噪声幅值
            height_measurements = 0.1

    # viewer camera:
    # 【分组】viewer：可视化相机初始位姿
    class viewer:
        # 【中文】相机跟随的环境索引
        ref_env = 0
        # 【中文】相机位置 x,y,z [m]
        pos = [10, 0, 6]  # [m]
        # 【中文】相机注视点 x,y,z [m]
        lookat = [11., 5, 3.]  # [m]

    # 【分组】sim：物理仿真器参数
    class sim:
        # 【中文】物理仿真步长[s]（1/0.005 = 200Hz）
        dt =  0.005
        # 【中文】每个仿真步内的子步数
        substeps = 1
        # 【中文】重力加速度向量[m/s^2]
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        # 【中文】世界坐标"上"方向：0 为 y 轴，1 为 z 轴
        up_axis = 1  # 0 is y, 1 is z

        # 【子分组】physx：PhysX 求解器参数
        class physx:
            # 【中文】CPU 求解线程数
            num_threads = 10
            # 【中文】求解器类型：0=PGS, 1=TGS（TGS 更稳定）
            solver_type = 1  # 0: pgs, 1: tgs
            # 【中文】位置迭代次数（越大接触越硬、越精确）
            num_position_iterations = 4
            # 【中文】速度迭代次数
            num_velocity_iterations = 0
            # 【中文】接触偏移[m]：提前生成接触的裕量
            contact_offset = 0.01  # [m]
            # 【中文】静止偏移[m]
            rest_offset = 0.0   # [m]
            # 【中文】反弹速度阈值[m/s]：低于此速度不反弹
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            # 【中文】最大去穿透速度
            max_depenetration_velocity = 1.0
            # 【中文】GPU 上最大接触对数量（环境越多需越大）
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            # 【中文】默认缓冲区大小的倍率
            default_buffer_size_multiplier = 5
            # 【中文】接触收集方式：0 不收集 / 1 仅最后子步 / 2 所有子步
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

# 【类说明】LeggedRobotCfgPPO：PPO 强化学习算法配置。
# 包含策略网络结构、PPO 训练超参数、运行/日志设置。
class LeggedRobotCfgPPO(BaseConfig):
    # 【中文】随机种子（-1 表示随机）
    seed = 1
    # 【中文】训练运行器类名（在注册表中按名字实例化）
    runner_class_name = 'OnPolicyRunner'
    # 【分组】policy：Actor-Critic 网络结构
    class policy:
        # 【中文】策略网络初始动作噪声标准差（探索强度）
        init_noise_std = 1.0
        # 【中文】Actor 网络隐藏层维度
        actor_hidden_dims = [512, 256, 128]
        # 【中文】Critic 网络隐藏层维度
        critic_hidden_dims = [512, 256, 128]
        # 【中文】激活函数
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    # 【分组】algorithm：PPO 算法超参数
    class algorithm:
        # training params
        # 【中文】价值函数损失的权重
        value_loss_coef = 1.0
        # 【中文】是否对价值损失也做 clip
        use_clipped_value_loss = True
        # 【中文】PPO 裁剪参数 epsilon（限制策略更新幅度）
        clip_param = 0.2
        # 【中文】熵奖励系数（鼓励探索）
        entropy_coef = 0.01
        # 【中文】每次迭代对同一批数据重复训练的轮数
        num_learning_epochs = 5
        # 【中文】每轮切分的 mini-batch 数量
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        # 【中文】学习率
        learning_rate = 1.e-3 #5.e-4
        # 【中文】学习率调度方式：adaptive 按 KL 自适应调整 / fixed 固定
        schedule = 'adaptive' # could be adaptive, fixed
        # 【中文】折扣因子 gamma（未来奖励衰减）
        gamma = 0.99
        # 【中文】GAE 的 lambda（偏差-方差权衡）
        lam = 0.95
        # 【中文】期望 KL 散度（adaptive 学习率的目标值）
        desired_kl = 0.01
        # 【中文】梯度裁剪范数上限
        max_grad_norm = 1.

    # 【分组】runner：训练运行与日志
    class runner:
        # 【中文】策略类名
        policy_class_name = 'ActorCritic'
        # 【中文】算法类名
        algorithm_class_name = 'PPO'
        # 【中文】每次迭代每个环境采集的步数（rollout 长度）
        num_steps_per_env = 24 # per iteration
        # 【中文】最大迭代次数（策略更新次数）
        max_iterations = 1500 # number of policy updates

        # logging
        # 【中文】每隔多少次迭代尝试保存一次模型
        save_interval = 50 # check for potential saves every this many iterations
        # 【中文】实验名（日志目录名）
        experiment_name = 'test'
        # 【中文】本次运行名
        run_name = ''
        # load and resume
        # 【中文】是否从检查点恢复训练
        resume = False
        # 【中文】加载哪一次运行：-1 表示最近一次
        load_run = -1 # -1 = last run
        # 【中文】加载哪个检查点：-1 表示最后保存的模型
        checkpoint = -1 # -1 = last saved model
        # 【中文】恢复路径（由 load_run 与 checkpoint 推导得到）
        resume_path = None # updated from load_run and chkpt