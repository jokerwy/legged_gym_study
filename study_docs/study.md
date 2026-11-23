# legged_gym 实战学习手册（study.md）

> 适用对象：已有强化学习理论基础（知道 MDP、策略梯度、PPO 是什么），已跑通过训练流程，但**还缺少系统调参、策略微调与工程交付经验**的学习者。
>
> 目标：通过本项目（legged_gym + rsl_rl + Isaac Gym，四足机器人运动控制）补齐"实战最后一公里"——会跑训练、会看曲线、会调参、会诊断问题、能讲清楚工程细节，达到强化学习相关岗位的面试与上手要求。
>
> 与工作区已有文档的关系（三份文档配合使用，不要重复读）：
>
> | 文档 | 定位 | 什么时候用 |
> |---|---|---|
> | `项目学习指南.md` | 原理手册：PPO 推导、并行仿真、奖励塑形、课程学习、域随机化 | 学某个概念时查 |
> | `代码精读指南.md` | 源码手册：逐文件逐函数讲解 | 读代码卡壳时查 |
> | **本文档** | **实战手册：按阶段做实验、调参、自测、面试准备** | 作为主线，从头到尾执行 |

---

## 目录

0. [学习路线图与时间安排](#0-学习路线图与时间安排)
1. [阶段一：跑通全流程（第 1–2 天）](#1-阶段一跑通全流程第-12-天)
2. [阶段二：读懂训练循环，把理论对应到代码（第 3–5 天）](#2-阶段二读懂训练循环把理论对应到代码第-35-天)
3. [阶段三：系统调参实战（第 6–12 天，核心阶段）](#3-阶段三系统调参实战第-612-天核心阶段)
4. [阶段四：进阶改造挑战（第 13–18 天）](#4-阶段四进阶改造挑战第-1318-天)
5. [阶段五：求职准备（贯穿全程）](#5-阶段五求职准备贯穿全程)
6. [附录 A：常用命令速查](#6-附录-a常用命令速查)
7. [附录 B：常见坑与排查](#7-附录-b常见坑与排查)
8. [附录 C：毕业自测清单](#8-附录-c毕业自测清单)
9. [全部思考题与自测参考答案](#9-全部思考题与自测参考答案)
10. [工程调参与策略微调实战](#10-工程调参与策略微调实战)

> **你现在的阅读顺序**：先读第 9 节补齐“能解释”，再按第 10 节完成“能独立做实验”。第 9 节给的是源码与理论答案；任何性能提升、最佳参数、消融收益，都必须由你自己的实验确认，不能直接当作简历成果。

---

## 0. 学习路线图与时间安排

总时长约 2–3 周（每天 2–4 小时）。**核心原则是"动手优先"**：每个知识点都必须对应一次实验或一段代码阅读，不允许只看不动手。

```
阶段一 跑通全流程        → 获得"我在训练机器人"的直观感受
阶段二 理论↔代码对照      → 把书上的 PPO 变成你读过的 200 行代码
阶段三 系统调参实战 ★     → 简历上"有调参经验"这句话的来源
阶段四 进阶改造挑战       → 证明你能改代码而不只是跑脚本
阶段五 求职准备           → 把项目经历讲成面试答案
```

每实际完成一个阶段，回到 [附录 C](#8-附录-c毕业自测清单) 打勾，再用第 10 节检验工程交付能力。清单是学习进度记录，不是岗位胜任或真机安全认证。

---

## 1. 阶段一：跑通全流程（第 1–2 天）

目标：不看原理，先把"训练 → 看曲线 → 回放策略 → 导出模型"这条流水线完整跑一遍，建立感性认识。

### 1.1 环境确认

```bash
conda activate legged_gym        # Python 3.8 + torch 2.0.1+cu118 + isaacgym
cd "/home/lixiang/AI Study/RL/legged_gym_study"
python -c "import torch; print(torch.cuda.is_available())"   # 应输出 True
```

如果环境有问题，先解决环境问题再继续（见附录 B）。**不要跳过验证直接训练**——调试一小时才发现 CUDA 不可用是新手最常见的浪费。

### 1.2 任务 1：启动第一次训练

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --headless
```

- `--task=anymal_c_flat`：ANYmal C 四足机器人在平地上学习跟踪速度指令
- `--headless`：不弹可视化窗口（训练更快；想看得去掉它）
- 当前平地任务默认 4096 个并行环境、**300 次迭代**；基类默认 1500 次，被平地子类覆盖。实际耗时取决于硬件和配置，不保证几十分钟内收敛。

**观察要点**（训练时的终端输出）：

1. `Mean reward` 是否从接近 0 稳步上升？
2. `Mean episode length` 是否从很小（机器人秒摔）变长？
3. 每个 iteration 大约几秒？（感受 GPU 并行的速度）

> 练习：用 `Ctrl+C` 中断训练，重新运行时加 `--resume --load_run <已有运行目录名> --checkpoint <编号>`，例如加载 `model_250.pt` 就传 `--checkpoint 250`，**不是传文件路径**。中断只保留此前已保存的 checkpoint，不会自动保存中断瞬间。当前续训并非逐位精确恢复，迭代计数也有已知限制，详见 10.4。

### 1.3 任务 2：用 TensorBoard 看曲线

```bash
tensorboard --logdir "/home/lixiang/AI Study/RL/legged_gym_study/logs" --port 6006
```

重点看这几组（由 `/home/lixiang/AI Study/RL/rsl_rl/rsl_rl/runners/on_policy_runner.py` 的 `log()` 方法写入）：

| 曲线组 | 含义 | 健康训练长什么样 |
|---|---|---|
| `Episode/rew_*` | 各奖励项加权后的回合累计值，除以配置的最大回合时长 | 任务类奖励（如 tracking）通常上升，负权重惩罚类趋近 0；需结合回合长度判断 |
| `Loss/value_function` | Critic 拟合误差 | 不持续失控、不出现 NaN；不要求始终下降 |
| `Loss/surrogate` | Actor 的 PPO 策略代理损失 | 允许正负波动，不能仅凭高低或正负判断训练效果 |
| `Policy/mean_noise_std` | 动作探索噪声 | 缓慢下降（学好了）或持平；归零过早=探索崩塌 |
| `Train/mean_reward` | 平均回报 | 长期趋势通常上升后进入平台期，允许短期波动 |

**这就是你今后调参时最主要的"仪表盘"，现在就混个脸熟。**

#### 评价指标

当前 `flat_anymal_c` 的训练日志共有 **21 个标量指标**，下面按 TensorBoard 中的实际名称解释。

##### 1. 训练效果：最先看这组

| 指标 | 含义 | 怎么看 |
|---|---|---|
| `Train/mean_reward` | 最近 100 个已结束回合的平均累计奖励 | 通常越高越好，反映综合表现 |
| `Train/mean_episode_length` | 最近 100 个已结束回合的平均长度，单位为环境步 | 变长通常说明更少提前摔倒；达到回合上限后，不再代表能力持续提高 |
| `Train/mean_reward/time` | 与 `mean_reward` 相同，但按训练耗时记录 | 用来比较相同时间内谁学得更好 |
| `Train/mean_episode_length/time` | 与 `mean_episode_length` 相同，但按训练耗时记录 | 用来观察稳定性随训练时间的变化 |

普通曲线横轴通常是**训练迭代次数**；这两个 `/time` 指标在代码中使用**累计训练耗时（秒）**作为记录步数。训练早期不足 100 个已结束回合时，取缓冲区中已有回合的平均值。

##### 2. 奖励细项：机器人哪些方面做得好？

**注意：`Episode/rew_*` 是加权后的奖励贡献，不是原始速度、力矩或误差。负权重的惩罚项通常为负值，越接近 0，说明惩罚越小。**

| 指标 | 简要解释 | 期望趋势 |
|---|---|---|
| `Episode/rew_tracking_lin_vel` | 实际前后、左右移动速度跟随目标速度的奖励 | 通常上升更好 |
| `Episode/rew_tracking_ang_vel` | 实际转向角速度跟随目标转向角速度的奖励 | 通常上升更好 |
| `Episode/rew_lin_vel_z` | 惩罚机身上下运动，抑制颠簸 | 接近 0 |
| `Episode/rew_ang_vel_xy` | 惩罚机身横滚、俯仰方向的快速摇晃 | 接近 0 |
| `Episode/rew_orientation` | 惩罚机身倾斜，鼓励保持水平 | 接近 0 |
| `Episode/rew_torques` | 惩罚关节力矩平方和，鼓励省力；不是直接测量耗电量 | 接近 0，但不能牺牲行走能力 |
| `Episode/rew_dof_acc` | 惩罚关节加速度过大，让运动更平滑；DOF 指关节自由度 | 接近 0 |
| `Episode/rew_action_rate` | 惩罚相邻两步动作变化过大，减少控制抖动 | 接近 0 |
| `Episode/rew_collision` | 惩罚配置指定的非足端部位碰撞 | 接近 0 |
| `Episode/rew_feet_air_time` | 在脚落地时，根据此前腾空时间给分，鼓励形成步态 | 结合实际步态判断，不是越高越好 |

`rew_feet_air_time` 有个特殊点：代码使用 **“腾空时间 − 0.5 秒”**计算落地得分，所以短于 0.5 秒也可能得到负值；移动指令很小时不计这一项。

这些奖励细项按**回合累计值除以配置的最大回合时长**记录，不是除以每个回合实际存活时间。因此，回合变长本身也会影响数值，不能只据此判断单步动作质量。

##### 3. 优化过程：网络是否正常学习？

| 指标 | 简要解释 | 怎么看 |
|---|---|---|
| `Loss/value_function` | Critic（价值网络）对未来回报的预测误差 | 关注长期失控、异常尖峰或 NaN；不要求始终下降 |
| `Loss/surrogate` | PPO 的策略代理损失，用于更新 Actor（动作策略） | 可以正负波动，**不能简单用“越低越好”判断训练效果** |
| `Loss/learning_rate` | 优化器当前学习率 | 看更新步幅如何变化；若采用自适应调度，变化通常是正常的 |
| `Policy/mean_noise_std` | 动作分布的平均标准差，表示探索噪声大小 | 常随学习下降，但不是越小越好；过早变小可能影响探索 |

强化学习的策略和数据分布不断变化，所以**损失曲线不像普通监督学习那样一定持续下降**。要和奖励、回合长度一起看。

##### 4. 运行性能：训练快不快？

| 指标 | 简要解释 | 怎么看 |
|---|---|---|
| `Perf/total_fps` | 所有并行环境合计每秒产生的环境步数，计算耗时包含采样和学习 | 越高通常越快；不是画面帧率 |
| `Perf/collection time` | 每次迭代采集仿真经验的耗时，单位秒 | 越低通常越快 |
| `Perf/learning_time` | 每次迭代更新神经网络的耗时，单位秒 | 越低通常越快 |

##### 5. 新手优先盯这 5 条

1. **`Train/mean_reward`**：综合表现有没有提高？
2. **`Train/mean_episode_length`**：有没有更少摔倒？
3. **`Episode/rew_tracking_lin_vel`**：有没有按目标速度走？
4. **`Episode/rew_orientation`**：机身有没有更平稳？
5. **`Perf/total_fps`**：训练速度是否正常？

**理想情况是：奖励提高、回合变长、速度跟踪改善，同时机身倾斜和碰撞惩罚减小。最后还要播放模型看实际动作——曲线好看，不一定走姿就好。**

### 1.4 任务 3：回放与导出

```bash
python legged_gym/scripts/play.py --task=anymal_c_flat
```

做三件事：

1. 观察机器人步态是否像"走路"（而不是抽搐、拖地、跳跃）；
2. 在 `/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/scripts/play.py` 搜索 `export_policy_as_jit`：策略导出为 TorchScript，可用于 C++ 推理，但导出不等于完成真机部署；
3. 通读该文件，理解它和训练入口的差异：为什么推理只需要 actor 不需要 critic？当前回放仍创建完整 runner 并加载 Actor-Critic checkpoint，只是不执行 critic 来选动作。没有显示环境时加 `--headless`，此时不会显示机器人窗口。

### 1.5 阶段一自测

- [ ] 能不看笔记说出一次训练的完整流程：配置 → 建环境 → 采数据 → PPO 更新 → 存 checkpoint
- [ ] 能说出 TensorBoard 里 5 条核心曲线分别反映什么
- [ ] 能解释为什么 4096 个环境并行能让训练从"数天"变"几十分钟"

---

## 2. 阶段二：读懂训练循环，把理论对应到代码（第 3–5 天）

目标：把你学过的 PPO 公式、Actor-Critic、GAE 全部落实到具体代码行上。**这一步是面试时"你读过源码吗"的底气。**

> 配套阅读：本阶段每个小节都标了源码位置，细节卡住时查《代码精读指南.md》对应章节。

### 2.1 任务 4：画出一次迭代的数据流

精读 `rsl_rl/rsl_rl/runners/on_policy_runner.py` 的 `learn()`（约第 92–180 行），在纸上画出一次 iteration 的数据流：

```
obs (4096, 48) ──actor──> actions (4096, 12) ──env.step──> rewards/dones/next_obs
     │                                                        │
     └──critic──> values (4096,)        存入 storage <────────┘
                                              │
                              compute_returns (GAE)
                                              │
                                    update(): 5 epoch × 4 minibatch
                                              │
                                     surrogate + value loss
```

**动手验证**：在 `on_policy_runner.py` 的循环里临时加两行 `print(obs.shape, actions.shape)`，跑一次（用 `--num_envs 64 --max_iterations 3` 快速跑），确认张量形状。亲手打印一次胜过读十遍。

### 2.2 任务 5：把 PPO 公式钉到代码上

打开 `rsl_rl/rsl_rl/algorithms/ppo.py` 的 `update()`（约第 120 行起），逐行找到下列公式对应的代码，并在笔记本上抄下"公式 ↔ 行号"对照表：

| 你学过的公式 | 代码位置（自己去定位） |
|---|---|
| 概率比 $r_t = \pi_\theta/\pi_{\theta_{old}}$ | `actions_log_prob - old_actions_log_prob` 再 `exp()` |
| clip 目标 $\min(rA, \text{clip}(r,1\pm\epsilon)A)$ | 本实现最小化负目标：`torch.max(surrogate, surrogate_clipped)` |
| 熵奖励 $\beta H(\pi)$ | `entropy_batch.mean()` |
| value loss（MSE 或 clip 版） | `value_loss = ...` |
| 自适应学习率（KL 触发） | `if kl_mean > desired_kl * 2.0 ...` |
| GAE $\hat A_t = \sum (\gamma\lambda)^l \delta_{t+l}$ | `compute_returns()`（约第 116 行） |

**关键思考题**（面试高频，务必能答）：

1. 为什么概率比用 log 相减再 exp，而不是直接算概率相除？（数值稳定性）
2. `use_clipped_value_loss` 开关是干什么的？为什么要 clip value？
3. 自适应学习率为什么以 KL 散度为信号？KL 大了说明什么？
4. 为什么采完数据后 `old_actions_log_prob` 就不能再变了？

### 2.3 任务 6：读懂环境的 MDP 定义

打开 `legged_gym_study/legged_gym/envs/base/legged_robot.py`，对照 `legged_robot_config.py`，回答四个问题（这是 RL 工程的"四要素"）：

1. **观测是什么**：`compute_observations()` 拼了哪些量？维度 48 是怎么来的？
2. **动作是什么**：`step()` 里 actions 如何变成力矩？（动作是目标关节角偏移，不是直接力矩；基类用 PD，当前 ANYmal 默认使用执行器网络，见 9.3）
3. **奖励是什么**：搜索 `_reward_*` 系列函数，各管什么？哪些权重为正、哪些为负？
4. **终止是什么**：`check_termination()` 里什么条件判 episode 结束？

**动手验证**：把 `compute_observations()` 里每个量用 `print` 打出 shape，加总确认 = 48。

### 2.4 任务 7：读懂配置继承体系

读 `envs/anymal_c/flat/anymal_c_flat_config.py`，回答：

- 它继承了哪两个基类？分别覆盖了哪些字段？
- 是否定义了 `__post_init__`？实际由哪个函数初始化嵌套配置？
- 为什么采用"基类默认 + 子类覆盖"而不是每个任务写全套配置？（这是工程上极常见的模式，面试可讲）

### 2.5 阶段二自测

- [ ] 能脱稿讲清 PPO 一次 update 的完整计算过程（storage → GAE → minibatch → 三个 loss → 自适应 lr）
- [ ] 能区分基类 PD 路径与 ANYmal 执行器网络路径，讲清动作缩放、控制频率及力矩限幅位置
- [ ] 能解释 GAE 中 $\lambda=0.95$ 时偏差/方差的权衡

---

## 3. 阶段三：系统调参实战（第 6–12 天，核心阶段）

目标：掌握 RL 工程师的核心技能——**提出假设 → 改变量做实验 → 看曲线验证 → 记录结论**。这是有 RL 理论但无实战的人最缺的部分，也是本手册的重心。

**实验纪律（每次实验必须遵守）**：

1. 一次只改一个变量（改了奖励权重又改 lr，曲线变了你不知道是谁的功劳）；
2. 用 `--experiment_name` 给每组实验起可辨识的名字，如 `--experiment_name lr_study`；
3. 每组实验跑 500–1500 iterations 即可下初步结论，不必等完全收敛；
4. 建一个 `experiments.md` 记录：假设、改动、结果（贴 TensorBoard 截图）、结论。这个文档日后就是你简历和面试的素材库；
5. 为省时间可先用 `--num_envs 1024` 或更少做探索性实验，结论稳定后再用 4096 验证。

### 3.1 实验一：PPO 超参数敏感性

依次做以下单变量实验（基线 = 默认配置），每组都在 TensorBoard 对比 `Train/mean_reward` 曲线：

| # | 改动 | 预期现象 | 验证的机理 |
|---|---|---|---|
| 1 | `learning_rate` 1e-3 → 1e-2 | 早期涨得快但很快震荡/崩掉 | lr 过大 → 策略更新步长超出信赖域 |
| 2 | `learning_rate` → 1e-5 | 几乎不学 | lr 过小 → 欠拟合 |
| 3 | `clip_param` 0.2 → 0.05 | 收敛变慢但更稳 | clip 小 = 更保守的策略更新 |
| 4 | `entropy_coef` 0.01 → 0.1 | 收敛慢、reward 上限略低，但曲线更平滑 | 熵正则 = 强制保持探索 |
| 5 | `entropy_coef` → 0.0 | `mean_noise_std` 快速塌缩，可能陷入局部最优 | 无探索压力 → 过早确定化 |
| 6 | `num_steps_per_env` 24 → 4 | 学习信号噪声大、value loss 高 | 批量太小 → 梯度方差大 |
| 7 | `lam`(GAE) 0.95 → 0.5 | 学得快但最终 reward 低 | λ 小 = 低方差高偏差 |

**产出要求**：画一张"超参数 → 收敛速度/最终性能"的总结表。这 7 组实验做完，你对 PPO 每个超参的"手感"就建立了。

### 3.2 实验二：奖励塑形（Reward Shaping）

奖励工程是 RL 落地最重要也最难的技能。在 `legged_robot_config.py` 的 `rewards.scales` 中做实验：

1. **去掉一个惩罚项**：把 `dof_acc` 权重设 0 → 观察机器人动作是否变得"抖动、抽搐"（高阶惩罚项管平滑性）；
2. **加大一个惩罚项**：`action_rate` ×10 → 机器人可能"僵住不动"（过度惩罚动作变化 → 保守到瘫痪）；
3. **总奖励截断消融**：当前 `only_positive_rewards` 已为 `True`，应对比 `True/False`；它把非终止项总和截断为非负，**不等于只保留任务奖励**。要只留任务奖励，需显式把其他权重设 0；
4. **改任务目标**：把 `tracking_sigma` 改小 → 跟踪奖励更“苛刻”，也可能使早期信号过弱；最终是否更准需要用原始跟踪误差验证。

**核心结论（自己用实验验证后再接受）**：奖励塑形决定了"学到什么行为"，超参数只决定"学多学快"。一个设计糟糕的奖励，什么超参都救不回来。

### 3.3 实验三：课程学习（Curriculum Learning）

1. 读 `legged_robot.py` 中 `_update_terrain_curriculum` / `update_command_curriculum`，说清：晋升到下一难度等级的条件是什么？
2. 实验：在 `anymal_c_rough`（混合地形）任务上，把 `terrain.curriculum` 设为 `False`（随机分配难度），对比开/关课程的 `Episode/rew_tracking_lin_vel` 和 terrain level 曲线。
3. 思考：课程学习和你学过的"模仿学习/预训练"有什么异曲同工之处？

### 3.4 实验四：域随机化与鲁棒性

1. 在 `legged_robot_config.py` 找到 `domain_rand` 段，逐项理解 `randomize_friction / push_robots / randomize_base_mass` 的物理含义；
2. 实验：分别关闭/开启摩擦随机化、推搡和质量随机化训练，再在相同固定低摩擦、载荷、推搡场景中比较。注意 `play.py` 会关闭摩擦随机化和推搡，但不关闭质量随机化；平面有效接触摩擦还有组合规则。不能直接把配置写成 0.1 就认定有效摩擦是 0.1 或已在训练分布之外，详见 10.6；
3. 思考：域随机化为什么有效？它和监督学习里的数据增强是不是一回事？

### 3.5 实验五：诊断一次"失败的训练"

故意制造失败，练诊断能力（挑 2 个做）：

- 把 `learning_rate` 调到 5e-2，观察 `Loss/surrogate` 单边漂移、`mean_reward` 崩掉 → 学会识别"发散"；
- 把任务奖励权重设为惩罚项的 1/100，观察机器人"宁可站着不动也不去跟踪速度" → 学会识别"奖励比例失衡"；
- 把 `max_iterations` 之前先用 `num_envs=8` 跑，观察曲线噪声极大 → 学会识别"样本不足"。

**每项写下：症状（哪条曲线什么形状）→ 原因 → 修复手段。** 这就是真实工作中的 debug 三板斧。

### 3.6 阶段三自测

- [ ] 给我任意一组训练曲线截图，我能猜出最可能的病因（lr 过大/探索崩塌/奖励失衡/样本不足）
- [ ] 能说出 clip、熵系数、GAE λ、batch size 各自调大调小的两个后果
- [ ] 能说清"奖励塑形决定学到什么，超参决定学多快"并用实验证据支持
- [ ] `experiments.md` 里有不少于 10 组有结论的对照实验

---

## 4. 阶段四：进阶改造挑战（第 13–18 天）

目标：从"会调参"升级到"会改代码"。按难度排序，能做多少做多少，前两个必做。

### 4.1 挑战一（必做）：新增一个奖励项

在 `legged_robot.py` 仿照 `_reward_orientation` 新增 `_reward_feet_slip`（惩罚着地时足端水平速度，让步态更稳）：

1. 写奖励函数（足端接触力 > 1N 时，惩罚足端水平线速度平方）；
2. 在 `rewards.scales` 注册权重；
3. 训练并对比足端滑移是否减少。

**这个挑战覆盖 RL 工程 80% 的日常改动：改观测/改奖励/注册配置。**

### 4.2 挑战二（必做）：训练 A1 机器人并对比

```bash
python legged_gym/scripts/train.py --task=a1 --headless
```

读 `envs/a1/a1_config.py`，回答：换机器人要改哪些配置？（URDF 路径、关节名、惩罚权重、初始姿态……）为什么观测维度不用改？**理解"框架如何把机器人差异收敛到配置文件"**。

### 4.3 挑战三（选做）：在观测里加入新信息

给观测向量加入足端接触标志（4 维），同时修改 `num_observations`，训练看是否改善。体会"观测设计 = 给策略的信息集"的含义。

### 4.4 挑战四（选做）：换一个地形组合

在 `utils/terrain.py` 的 `terrain_utils` 里组合新的地形（如加长楼梯），在配置里注册，训练看课程学习如何带机器人爬楼。

### 4.5 挑战五（进阶选做）：消融自适应学习率

把 PPO 的学习率改为固定 3e-4（去掉 KL 自适应逻辑），重跑基线，量化自适应机制带来的收益。**这是论文级的消融实验思维。**

---

## 5. 阶段五：求职准备（贯穿全程）

### 5.1 把项目写进简历（示例措辞）

> 基于 Isaac Gym + legged_gym + rsl_rl 复现四足机器人速度跟踪策略（PPO，4096 并行环境）；系统性完成 10+ 组超参数/奖励塑形/域随机化消融实验，沉淀调参方法论；新增足端滑移奖励项并完成训练验证；熟悉 GPU 大规模并行 RL 训练流水线与 Sim-to-Real 关键技术（课程学习、域随机化、JIT 策略导出）。

> 上面是**完成相应实验之后**的示例，不能直接作为你目前的经历。现在可先写“跑通四足机器人 PPO 仿真训练与回放流程，学习配置继承、奖励分项与训练日志诊断”；其余内容做完并有证据后再补充。JIT 导出与理解 Sim-to-Real 不等于完成真机部署。

### 5.2 高频面试题（用本项目素材回答）

**PPO 算法类**：

1. PPO 相比 TRPO 简化了什么？clip 的作用是什么？→ 用 `ppo.py` 里负目标的 `torch.max(surrogate, surrogate_clipped)` 讲
2. GAE 是什么？λ 和 γ 分别控制什么？→ 用 `compute_returns()` 讲
3. 为什么要 advantage 归一化？代码里在哪做的？→ 在 `rollout_storage.py` 的 `compute_returns()` 中找
4. Actor-Critic 里 critic 只用来训练、部署时丢弃，为什么可以？→ 回放只用 actor 选动作，导出前馈模型只含 actor

**工程类**：

5. 4096 个环境并行，数据怎么组织？为什么整个 pipeline 不放回 CPU？→ Isaac Gym GPU 张量直出，避免 PCIe 拷贝
6. 训练 reward 不涨你会怎么排查？→ 看曲线提出发散/失衡/欠采样假设，再用原始指标与单变量实验验证，见 9.5
7. 仿真训练的策略怎么上真机？gap 在哪？→ 域随机化 + 执行器建模/PD 控制抽象 + JIT 导出 + 状态估计/安全控制
8. 观测里为什么要放历史/为什么要归一化？→ `obs_scales`、`clip_observations` 的作用

**项目类**：

9. 你调过最有效的参数是什么？→ 讲你的 3.1 节实验表里印象最深的两组对照
10. 你改过最有意思的东西是什么？→ 讲挑战一新增奖励项的完整过程（需求→实现→验证）

### 5.3 建议的知识补强（项目之外）

本项目是 on-policy 流派，求职还需口头覆盖：

- DQN 系列（经验回放、target network）
- SAC/TD3（off-policy、最大熵框架）—— 与 PPO 的适用场景对比是必考题
- 模仿学习 / Offline RL 的基本概念
- 机器人方向：Sim-to-Real 文献（本项目论文 arXiv:2109.11978 精读）

---

## 6. 附录 A：常用命令速查

```bash
cd "/home/lixiang/AI Study/RL/legged_gym_study"

# 平地训练（当前默认 4096 环境 / 300 迭代）
python legged_gym/scripts/train.py --task=anymal_c_flat --headless

# 快速冒烟实验（几分钟出结果）
python legged_gym/scripts/train.py --task=anymal_c_flat --headless \
    --num_envs 512 --max_iterations 200 --experiment_name smoke --run_name test1

# 断点续训
# 先填写已有运行目录名；编号 250 对应 model_250.pt，必须确实存在
SOURCE_RUN='填写已有运行目录名'
python legged_gym/scripts/train.py --task=anymal_c_flat --headless --resume \
    --experiment_name flat_anymal_c --load_run "$SOURCE_RUN" --checkpoint 250 \
    --max_iterations 100 --run_name resume_100

# 看曲线
tensorboard --logdir "/home/lixiang/AI Study/RL/legged_gym_study/logs" --port 6006

# 回放 + 导出 JIT 策略（无显示环境加 --headless；有窗口则去掉它）
python legged_gym/scripts/play.py --task=anymal_c_flat --headless

# 可用任务列表
python -c "import isaacgym; from legged_gym.envs import *; from legged_gym.utils import task_registry; print(list(task_registry.task_classes.keys()))"
```

超参数修改位置一览：

| 想改什么 | 改哪里 |
|---|---|
| PPO 超参（lr/clip/熵/GAE） | `/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/envs/base/legged_robot_config.py` 的 `LeggedRobotCfgPPO.algorithm` 或子类覆盖 |
| 奖励权重 | config 的 `rewards.scales` |
| 网络结构（隐藏层/激活） | 训练 config 的 `policy`（`actor_hidden_dims` 等） |
| rollout 长度/迭代数 | 训练 config 的 `runner.num_steps_per_env / max_iterations` |
| 观测/动作维度 | config 的 `env.num_observations / num_actions` |
| 课程/地形 | config 的 `terrain` / `commands` |
| 域随机化 | config 的 `domain_rand` |

---

## 7. 附录 B：常见坑与排查

1. **nvrtc arch 错误 / kernel 无法编译**：RTX 4060 是 sm_89，torch 1.10+cu113 不支持，必须用 torch 2.0.1+cu118（本环境已配好，不要乱动 torch 版本）。
2. **训练时显存爆掉（OOM）**：减 `--num_envs`，4096 → 2048 → 1024；同时总 batch 和吞吐会改变，不能保证性能影响小，需按交互步数和耗时重新比较。
3. **训练卡住不动、CPU 占用高**：确认没去掉 `--headless` 时窗口是否正常渲染；远程/无显示环境必须 `--headless`。
4. **TensorBoard 没有新曲线**：训练日志默认写在 `legged_gym_study/logs/<experiment>/<日期时间>_/`，确认 `--logdir` 指对层级。
5. **改了配置没生效**：优先检查你改的是不是基类被子类覆盖了——子类 config 的字段会覆盖基类（去任务自己的 config 文件确认）。
6. **训练结束后机器人表现差**：先看是不是根本没收敛（迭代数不够），再用 play.py 回放观察具体失败模式（摔/僵/漂），对应回 3.5 节的诊断表。

---

## 8. 附录 C：毕业自测清单

**工程能力**

- [ ] 能独立启动/中断/续训/回放一次完整训练
- [ ] 能画出一次 iteration 的完整数据流（含张量形状）
- [ ] 能定位任意一个超参数/奖励权重在配置体系中的位置并修改生效
- [ ] 能新增一个奖励项并训练验证

**算法理解**

- [ ] 能脱稿推导 PPO clip 目标并指出对应代码行
- [ ] 能讲清 GAE、advantage 归一化、KL 自适应学习率各自解决什么问题
- [ ] 能解释为什么 on-policy 算法吃并行环境红利最大

**调参实战**

- [ ] 完成阶段三全部实验并有书面记录（≥10 组对照）
- [ ] 能结合曲线、原始指标、回放提出发散/探索不足/奖励失衡的假设，并通过对照实验验证，而非仅凭截图断言病因

**求职输出**

- [ ] 简历上有这个项目的一段合格描述
- [ ] 5.2 节 10 道面试题全部能流利作答
- [ ] 精读项目论文 arXiv:2109.11978，能讲清"为什么分钟级训出可上真机的步态"

全部打勾 = 你已经从“学过 RL”向“做过 RL”迈进；真实岗位还需以下工程实践。读懂答案不等于完成实验，不要提前打勾。

---

## 9. 全部思考题与自测参考答案

本节按原章节对应作答，包括思考题、配置阅读问题、阶段自测和 10 道面试题。源码位置用函数名定位，避免注释增加后行号失效。

### 9.1 阶段一：完整流程、核心曲线、并行与推理

**Q1：一次训练的完整流程是什么？**

1. 导入任务包完成注册，解析 CLI，取得环境/训练配置，应用支持的命令行覆盖并设随机种子。
2. 建立地形、机器人与并行仿真，初始化观测、接触、动作等 GPU 缓冲区。
3. 创建 Actor-Critic、Adam、PPO、RolloutStorage、日志目录；如果续训则加载 checkpoint。
4. 固定本轮采样策略，每个环境采集 `T` 步：观测 → Actor 采样动作，Critic 估值 → 仿真 → 奖励/终止 → 写 storage。
5. 计算末尾状态价值，反向递推 GAE 和回报，归一化优势。
6. 对本轮数据做多 epoch、小批量 PPO 更新，写日志、定期保存；清空 rollout，再采新数据。
7. 用固定测试场景评估候选 checkpoint，检查步态，再导出 Actor。训练 reward 最高不自动等于部署最优。

**Q2：能说出五条核心曲线吗？**

- `Train/mean_reward`：最近结束回合的累计奖励；只能在奖励定义相同的条件下直接比较。
- `Train/mean_episode_length`：存活环境步数；长通常更稳定，达到上限后要看任务质量。
- `Loss/value_function`：回报拟合误差；目标随策略变动，不要求单调下降。
- `Loss/surrogate`：最小化的 PPO 负代理目标；正负或大小不能单独衡量策略好坏。
- `Policy/mean_noise_std`：随机动作分布标准差；表示训练探索，不是部署动作必然有噪声。

也可补充跟踪奖励、姿态惩罚、FPS。奖励项的日志是**加权、累计并按最大回合时长缩放**的值，不是原始 m/s、N·m 或能耗。

**Q3：为什么 4096 并行环境会快？为什么 on-policy 特别受益？**

GPU 同时推进大量机器人，Actor 也以 batch 推理；仿真状态可直接包装成 GPU 张量，避免每步搬整批状态过 PCIe。PPO 要反复采当前策略的新数据，不能像典型 off-policy 算法一样长期利用 replay buffer，所以提高采样吞吐非常重要。但加速不是 4096 倍：碰撞求解、显存、优化与同步都有开销，4096 环境也不是 4096 个独立训练进程。日志仍有 CPU 传输，不能说“全流程绝不回 CPU”。

**Q4：为什么回放/部署只需要 Actor？**

Actor 实现 `obs → action`；Critic 用于构造训练优势和价值目标，部署选动作不依赖它。当前 `get_inference_policy()` 返回 `act_inference`，输出高斯策略均值而非采样。回放脚本仍创建完整 runner；JIT 导出的前馈模型才仅含 Actor，不含观测预处理、PD、执行器仿真或安全层。

### 9.2 阶段二：PPO 四道思考题与一次 update

源码：`/home/lixiang/AI Study/RL/rsl_rl/rsl_rl/algorithms/ppo.py` 的 `act()`、`update()`；GAE 在 `/home/lixiang/AI Study/RL/rsl_rl/rsl_rl/storage/rollout_storage.py` 的 `compute_returns()`。

**Q1：概率比为什么用 log 相减再 exp？**

多维高斯联合概率密度是各维密度的乘积，12 维直接相乘可能很小，分别计算分子分母再相除容易下溢。先求各维 log-density 并求和，再用 `exp(logp_new - logp_old)` 得到同一比值，数值上更稳，也方便反向传播。连续动作这里是**密度比**，不是某个精确动作的概率。极端 log 差仍会导致 `exp` 溢出；还要检查标准差、观测、梯度和学习率。

**Q2：`use_clipped_value_loss` 干什么？为什么 clip value？**

令 $V_{old}$ 是采样时的价值，$R$ 是固定回报目标：

$$V_{clip}=V_{old}+\operatorname{clip}(V_\theta-V_{old},-\epsilon,\epsilon)$$
$$L_V=\mathbb E[\max((V_\theta-R)^2,(V_{clip}-R)^2)]$$

这样避免 Critic 因同一批数据多次更新而过度改变估值；关闭则使用普通 MSE。这里裁剪的是**相对旧价值的变化**，不是把价值强制限制在 `[-0.2, 0.2]`；也不是硬保证新价值变化不超过阈值。本实现复用 `clip_param`，数值意义受奖励尺度影响，奖励改变很大时可能让 Critic 适应偏慢。

**Q3：为什么以 KL 为自适应学习率信号？KL 大说明什么？**

KL 测量在本批观测上旧/新动作分布的偏离，比参数距离更贴近行为变化。本实现用对角高斯解析式计算 $D_{KL}(\pi_{old}\Vert\pi_{new})$ 的 batch 均值：大于 `2 * desired_kl` 就把 lr 除以 1.5，低于一半且大于零则乘 1.5，限制在 `[1e-5, 1e-2]`。KL 大意味着更新可能过猛、旧数据与新策略不够接近，**不等于已证明任务性能下降**。这是启发式调速，不是 TRPO 的严格约束，也不是 KL early stopping。

**Q4：为什么采完数据后 `old_actions_log_prob` 不能变化？**

它是行为策略产生这些动作时的固定基准。如果每个 epoch 都用当前策略重算分母，ratio 会趋于 1，PPO 无法知道新策略相对采样策略改变了多少。固定的不仅是 old log-prob，还包括采样动作、旧价值、旧均值/标准差及这轮 GAE 目标。更新的是**新策略对同一动作的 log-prob**；下一轮才换采样策略。

**Q5：一次 update 的公式与数据流？**

平地默认 `N=4096, T=24`：`obs=[N,48]`，`actions=[N,12]`，value/log-prob 的 storage 形状为 `[T,N,1]`，观测为 `[T,N,48]`。`rewards/dones` 从环境返回时是 `[N]`。本轮总样本 `B=98304`，4 个 minibatch 时每批 24576，5 epoch 共 20 次优化：

1. 先按时间反向计算 GAE，再展平时间/环境维；不能先随机打乱轨迹再算 GAE。
2. 用当前 Actor 对**存下来的动作**算 log-prob、熵与均值/标准差，Critic 算新 value。
3. KL 自适应调 lr；计算 ratio、负 clipped surrogate、value loss、负熵正则。
4. `loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()`。
5. `zero_grad → backward → clip_grad_norm_ → optimizer.step`，最后清空 storage。

clip 的“保守”来自对有利方向的收益截断：$A>0$ 时不再奖励过大的 ratio，$A<0$ 时不再奖励过小的 ratio。它**不直接裁剪动作、不直接限制每个参数，也不能保证 KL 总不超标**。

**Q6：GAE、γ、λ、优势归一化各解决什么问题？**

$$\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)$$
$$\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},\qquad R_t=\hat A_t+V(s_t)$$

- `γ` 控制未来奖励折扣，也会影响任务目标；越接近 1，关注的未来越远，估计通常更难。
- `λ` 混合不同长度 TD 估计。接近 0 更依赖 Critic，通常低方差、高偏差；接近 1 更接近长程回报，通常高方差、低 bootstrap 偏差。`λ=0.95` 是常用折中，不是普适最优。
- 优势表示“比基线好多少”，有助降低梯度方差。`compute_returns()` 中对**整轮 rollout**执行 `(A - mean) / (std + 1e-8)`，稳定 Actor 更新尺度；不是把 Critic 的回报也归一化，不保证所有奖励尺度问题消失。
- 真正终止不 bootstrap；时间截断应保留未完成任务的价值。当前旧实现 `process_env_step()` 用 `transition.values` 补 timeout 奖励，即采样当前状态的 $V(s_t)$，**不能把它说成精确使用终止前的 $V(s_{t+1})$**。严格处理需额外保存 terminal observation，避免拿自动 reset 后的观测估值。

### 9.3 阶段二：环境 MDP 四要素

**Q1：48 维观测由什么组成？**

| 切片 | 内容 | 维度 |
|---|---|---|
| `[0:3]` | 机体系基座线速度，乘 `lin_vel` scale | 3 |
| `[3:6]` | 机体系基座角速度，乘 `ang_vel` scale | 3 |
| `[6:9]` | 重力单位向量在机体系的投影 | 3 |
| `[9:12]` | x/y 线速度与 yaw 角速度指令，乘 command scales | 3 |
| `[12:24]` | 关节位置减默认站姿角，乘 scale | 12 |
| `[24:36]` | 关节速度，乘 scale | 12 |
| `[36:48]` | 刚执行的动作，对下一次决策而言是历史动作 | 12 |

总和 `3+3+3+3+12+12+12=48`。平地关闭高度测量；rough 追加 `17×11=187` 个高度，共 235 维。指令缓冲区有 4 维，但 heading 不直接拼进观测。机体坐标表示降低对世界位置/朝向的依赖；实际真机线速度需要状态估计，不能直接使用仿真真值。

**Q2：动作为什么是关节角偏移？如何变成力矩？**

12 维动作对应 12 个驱动关节，目标为 $q_{des}=q_{default}+s_a a$。位置目标更接近底层关节控制接口，利用已有伺服控制，通常比让策略从零学习直接力矩更易稳定，但仍依赖执行器模型准确性。

- **基类/A1 的 P 路径**：采样动作 → `clip_actions` → `action_scale` → $\tau=K_p(q_{des}-q)-K_d\dot q$ → 按 URDF 力矩上限裁剪 → 施加仿真力矩。
- **当前 ANYmal 的默认路径**：注册的是 `Anymal` 子类，继承 `use_actuator_network=True`。`/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/envs/anymal_c/anymal.py` 重写 `_compute_torques()`，把位置误差和关节速度送进预训练 LSTM 执行器网络输出力矩，**不是直接使用上面的 PD 公式**。该分支返回前没有基类的显式 `torch.clip`，不能照搬“所有路径都有力矩 clip”的结论，部署必须另设安全限幅。
- 默认仿真 `dt=0.005s`、`decimation=4`，策略周期为 0.02s（50Hz），物理/力矩计算 200Hz。同一动作保持 4 个仿真步，每个物理步重新计算力矩。不是每个物理步都重新采样策略动作。

**Q3：奖励各管什么？哪些正、哪些负？**

当前平地有效权重（包括继承项）：

| 奖励项 | 作用 | 配置权重 |
|---|---|---|
| `tracking_lin_vel` | `exp(-xy速度误差平方和 / tracking_sigma)` | `+1.0` |
| `tracking_ang_vel` | yaw 角速度跟踪指数奖励 | `+0.5` |
| `lin_vel_z` | 抑制上下速度平方 | `-2.0` |
| `ang_vel_xy` | 抑制 roll/pitch 角速度平方和 | `-0.05` |
| `orientation` | 抑制重力投影 xy 平方和（倾斜） | `-5.0` |
| `torques` | 力矩平方和，不是实际电耗 | `-0.000025` |
| `dof_acc` | 相邻速度差除 dt 后的平方和 | `-2.5e-7` |
| `action_rate` | 相邻动作差平方和，不含除 dt | `-0.01` |
| `collision` | 指定惩罚部位接触力模长 >0.1N 的数量 | `-1.0` |
| `feet_air_time` | 首次落地时奖励腾空时间减 0.5s，非零移动指令门控 | `+2.0` |

其余已实现函数覆盖高度偏差、关节速度、位置/速度/力矩软超限、绊脚、静止指令下关节偏移、过大接触力、失败终止等。**有函数不等于该任务已启用**：零权重项在 `_prepare_reward_function()` 移除；例如 `max_contact_force=350` 只是阈值，没有启用对应权重就不会产生惩罚。

有效权重初始化时乘策略 `dt`。普通奖励加总后，`only_positive_rewards=True` 才把总和下限截为 0，然后另加终止奖励。正权重不保证原始函数总为正，例如短腾空的 `feet_air_time`。不要只看权重绝对数值，平方力矩与指数奖励量纲、原始尺度不同。

**Q4：什么时候终止？**

`check_termination()`：指定终止接触部位的接触力模长任一大于 1N，或 `episode_length_buf > max_episode_length`；两者 OR 后 reset。默认终止部位是 base。这里没有另写“倾角超过阈值就终止”。失败与 timeout 要分开统计，时间上限结束不是摔倒。

### 9.4 阶段二：配置继承三个问题

**Q1：平地配置继承谁，覆盖什么？**

两条链分别是 `AnymalCFlatCfg → AnymalCRoughCfg → LeggedRobotCfg → BaseConfig` 与 `AnymalCFlatCfgPPO → AnymalCRoughCfgPPO → LeggedRobotCfgPPO → BaseConfig`。

- 环境覆盖：观测 48、平面、关闭测高、启用自碰撞、接触力阈值 350、orientation/torques/feet_air_time 权重、直接 yaw 指令、4s 重采样、yaw 范围 `[-1.5,1.5]`、摩擦随机范围 `[0,1.5]`。
- 训练覆盖：Actor/Critic `[128,64,32]`、ELU、熵系数 0.01、实验名 `flat_anymal_c`、run/load 设置、**max_iterations=300**。
- 未覆盖项沿父类，例如 4096 环境、24 rollout 步、5 epoch、4 minibatch、学习率自适应、质量随机化，以及 ANYmal 执行器网络。

**Q2：`__post_init__` 做什么？**

本项目这些类不是 dataclass，没有此方法。实际是 `/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/envs/base/base_config.py` 的 `BaseConfig.__init__()` 调 `init_member_classes()`，递归把嵌套类变为实例。不能把别的框架的初始化机制套到这里。

**Q3：为什么用“默认 + 子类覆盖”？怎样确认改动生效？**

复用通用参数、集中表达任务差异、减少复制和配置漂移；代价是需要追踪继承链、嵌套类也需继承父配置分组。最终读取顺序大致是基类 → 子类 → 脚本直接赋值/支持的 CLI 覆盖（具体以调用顺序为准）。检查**创建环境和 runner 前实际配置对象**，而不是只看改过的文件。注册表返回的是已注册配置对象，批量实验宜使用独立进程或深拷贝，避免前次修改污染后次。

### 9.5 阶段三：调参、课程学习、域随机化与诊断

**Q1：clip、熵系数、GAE λ、batch 调大调小有什么后果？原实验现象一定出现吗？**

| 参数 | 调大可能怎样 | 调小可能怎样 | 不能忽略的条件 |
|---|---|---|---|
| learning rate | 更新更快，但 KL/梯度尖峰、性能退化风险增加 | 更慢、更保守，也可能适应不足 | 默认 adaptive 会改写初始 lr，纯 lr 对照需 fixed |
| clip ε | 代理目标允许更激进的有利变化，可能加快也可能不稳 | 有利变化更早饱和，可能更稳也可能学不动 | 不是严格信赖域；value clip 也复用 ε |
| entropy coefficient | 鼓励保留随机性，探索更足，可能妨碍精细步态 | 更易集中到当前解，但可能探索不足 | 设 0 不代表必定噪声立即塌缩 |
| GAE λ | 更看重长程 TD 残差，通常低偏差、高方差 | 更依赖 value，通常低方差、高偏差 | Critic 质量与奖励延迟影响很大 |
| 总 batch `N×T` | 梯度估计通常更稳、每轮成本增大，更新频率降低 | 更新更频繁，但噪声更大 | 需要同时报告样本与时间，不只比 iteration |
| rollout T | 保持 N 时样本更多、长程信息更足 | 样本更少、GAE 更依赖末尾 bootstrap | 改 T 不只是改 batch，还改截断长度 |
| epochs | 同批数据利用更充分，但过拟合旧数据/KL 过大风险增加 | 保守，但可能没充分利用本轮样本 | PPO 不能无限重复旧 rollout |

3.1 的“涨得快/必崩/最终更低”都是待检验假设，不是标准实验结果。`lr=1e-5` 不必然完全不学；`λ=0.5` 不必然先学快后变差。正确产出是实际曲线与固定测试指标、多种子统计，再解释机制。

**Q2：奖励塑形为什么决定行为？三个惩罚/跟踪实验如何解释？**

策略优化的是你定义的奖励，不是自然语言里的“走得像机器人”。去掉关节加速度惩罚可能让抖动更便宜；动作变化惩罚过强可能使不动比跟踪划算；缩小 tracking sigma 可能使好/坏跟踪更易区分，也可能让大误差区域几乎全零。`only_positive_rewards` 的正确含义见 9.3。

“奖励决定学到什么、超参决定学多快”是经验简化：奖励定义优化目标，超参影响优化与探索，**也会影响最终能找到哪个局部解**。奖励好但探索崩溃一样失败；严重奖励漏洞一般不能靠 lr 修复。最可靠验证是原始速度误差、抖动、滑移及摔倒率，而不是新奖励下总 reward 变高。

**Q3：地形/指令课程的晋升条件是什么？和模仿学习/预训练有什么相似不同？**

- 地形课程在 reset 时更新：水平位移超过半块地形长度升级；未升级且距离不到“当前指令 xy 模长 × 最大回合秒数 × 0.5”则降级。达到最高等级后随机分配到已有等级，而非永远停在最高级。这里只用净位移，不是累计路径长度或精确跟踪误差。
- 指令课程：已结束环境的线速度跟踪累计奖励除最大回合步数，超过 `0.8 * tracking_lin_vel有效权重` 时，把 x 速度范围两端各扩 0.5，受 `max_curriculum` 限制。
- 平地任务解析配置时关闭地形课程，指令课程基类默认 False。想做课程实验，先确认相应任务真正启用。rough 地形等级日志为 `Episode/terrain_level`。
- 相似处：都让学习从更容易/更有帮助的起点开始，减少探索难度。不同处：课程改变任务/数据分布；模仿学习用专家动作监督；预训练先获得可迁移参数。不要求课程使用专家，也不要求预训练由简单到难。

**Q4：域随机化的三项物理含义、为什么有效、与数据增强有何异同？**

摩擦随机化改变足底接触条件；质量随机化改变基座附加质量、运动响应与负载；push 在代码中是**覆盖基座 xy 速度**模拟瞬时扰动，并不是持续施加某个牛顿数的外力。摩擦/质量通常建环境时采样，不应说每个物理步都重采样；push 按时间触发，观测噪声则按步加入。

可理解为优化 $\mathbb E_{\phi\sim p(\phi)}[J(\pi;\phi)]$，逼策略在一组环境参数下都取得可接受表现，降低对单一仿真器参数的过拟合。类似数据增强的共同点是增加训练分布多样性；不同点是 RL 中随机化动力学还改变状态转移、接触及未来轨迹，不只是变换静态输入。它通常改善**所覆盖范围附近**的鲁棒性，不保证任意 OOD 泛化，也不等于优化最坏工况安全性。范围过大还可能使任务不可学，需标定合理范围和随机化课程。

**Q5：失败训练怎么诊断？能凭曲线截图秒判吗？**

不能唯一判因。先提出假设，再做最低成本验证：

| 症状 | 优先假设 | 验证与修复 |
|---|---|---|
| reward 突降、value 尖峰/NaN | 更新过猛、输入异常、物理失稳 | 检查有限值、实际 lr、KL、梯度、动作饱和；从上个好模型降 lr 重试 |
| 存活变长但不追速度 | 指令接近零、惩罚压过任务、动作缩放/接口错误 | 固定非零指令，查原始速度误差、目标角、力矩与分项贡献 |
| noise 很低且 reward 卡住 | 探索不足或已收敛 | 换测试任务确认；适度增熵/探索，检查 std 是否有限且为正 |
| 曲线很抖、种子差异大 | batch 小、环境多样性/课程分布变化 | 对齐样本预算，增加 N/T，重复种子，检查地形等级与命令分布 |
| 训练好、回放或新场景差 | 测试分布变了、checkpoint 错了、策略过拟合 | 核对模型路径及噪声/摩擦/质量开关，分桶测试，不只看默认 play |
| reward 高但动作拖地/乱跳 | 奖励漏洞或质量指标缺失 | 加入原始滑移/接触/抖动评估，调整奖励并做消融 |

现有 TensorBoard 默认**没有独立 KL、clip fraction、梯度范数、原始 RMSE**曲线，诊断这些需要新增日志或断点检查，不能假装它们已经存在。超参修复前先排除配置没生效和动作/坐标系错误。

### 9.6 阶段四：五个改造挑战的参考答案

**Q1：如何新增足端滑移奖励？**

定义 $p_{slip}=\sum_{foot}\mathbf 1[F_z>1N]\|v^{world}_{foot,xy}\|^2$，函数返回 `[num_envs]` 的非负惩罚量，配置用负权重。静止地面下应惩罚**足端相对地面的世界系速度**，不要误减基座速度；移动平台需减平台速度。当前基类没有保存足端 rigid-body 速度，不能直接假设已有 `self.feet_vel`。

完整改动链：

1. 在 `_init_buffers()` 用 Isaac Gym `acquire_rigid_body_state_tensor()` 取得状态并 `gymtorch.wrap_tensor(...).view(num_envs, num_bodies, 13)`，保存到新属性；速度在最后一维 `[7:10]`。
2. 在 `post_physics_step()` 计算奖励**之前**调用 `refresh_rigid_body_state_tensor()`，确保不是上一帧速度。
3. 新增 `_reward_feet_slip()`，取足端索引、`[7:9]` 水平速度与接触 mask，沿足数维累加。浮点 dtype/设备和其他奖励一致，不逐机器人 Python 循环。
4. 在目标任务 `rewards.scales` 添加负权重 `feet_slip`；注册依靠 `_reward_<名称>` 自动绑定，不需改 PPO。
5. 测试：零速接触→0，摆动不接触→0，接触滑动→正；shape、有限值正确。固定评估看每接触步滑移均值/P95、跟踪误差及摔倒率，防止“更少走路，所以更少滑”的伪提升。

接触传感器在三角网格上有噪声/局限，阈值与滤波需验证。新增奖励并不意味着肯定提高性能，权重必须扫描。这一挑战此处提供实现路线，**没有替你改训练源码或声称已训练验证**。

**Q2：换 A1 要改哪些配置？为什么维度可以不变？**

需检查 URDF/网格/惯量/关节限位、关节和足端名称、默认站姿与基座高度、Kp/Kd、action scale、decimation、碰撞惩罚/终止部位、奖励尺度、随机化范围及任务注册类。A1 配置实际用 `LeggedRobot`、PD `Kp=20, Kd=0.5`、scale 0.25；ANYmal 用 `Anymal`、执行器网络、scale 0.5。不能只换 URDF。

两者都是 12 个驱动关节，使用同一组本体观测，故相同测高设置下维度相同；**默认 a1 是 rough，为 235 维，不是平地 ANYmal 的 48 维**。相同维度不意味着关节顺序/动作语义相同，不能直接加载模型就认为可迁移。换成不同 DOF 的机器人还要改观测、动作、硬编码噪声切片等。

**Q3：加 4 维接触观测具体还要改什么？**

选择接触定义与滤波，追加 4 维标志，平地观测改 52、rough 改 239；同步 actor/critic 输入、`num_privileged_obs`（若使用）、噪声向量切片、reset 与部署观测构造。当前高度噪声切片硬编码 `[48:235]`，若接触插在高度前必须移动，追加在最后则需明确其噪声为零或另设模型。新输入维度让旧 checkpoint 第一层不兼容；`strict=False` 也不会自动解决同名参数 shape mismatch，需显式迁移或重训。

**Q4：换地形要注意什么？**

`/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/utils/terrain.py` 的 `Terrain` 负责生成和拼接，`terrain_utils` 是调用的 Isaac Gym 工具模块，不是随意添加配置字段就能自动注册地形。要改具体生成分支、类型比例、难度参数映射，校验高度量化、楼梯尺寸、出生区域、边界、碰撞与课程 origin。先可视化/低环境冒烟，再训练，固定每种类型/难度评估，避免平均值掩盖某种地形全失败。

**Q5：消融自适应 lr 怎么做才公平？**

不必删除 PPO 代码：配置 `schedule='fixed', learning_rate=3e-4` 即可关闭现有 KL 分支。比较 adaptive 和 fixed 时固定其他参数、种子、样本预算与评估场景；最好再有 fixed `1e-3` 对照，避免把调度收益与初始 lr 差异混为一谈。报告达到指标阈值所需样本/时间、最终 RMSE/摔倒率、方差，以及实际 lr 轨迹。没有实测结果时答案是实验设计，不是“自适应必定更优”。

### 9.7 阶段五：10 道高频面试题逐题答案

**1. PPO 比 TRPO 简化了什么？clip 有什么作用？**

TRPO 近似求解带 KL 约束的策略优化，常涉及 Fisher/Hessian-vector product、共轭梯度与线搜索；PPO 用易实现的 clipped surrogate 和普通一阶优化器，多次小批量更新。clip 截断过度有利的概率比变化，提供保守更新的启发，不是严格保证单调提升。结合本项目 `torch.max` 的负目标讲，具体公式见 9.2。

**2. GAE 是什么？γ 与 λ 控制什么？**

GAE 是指数加权的多步 TD 残差优势估计，公式见 9.2。γ 控制奖励折扣/任务未来视野，λ 控制 bootstrap 偏差与采样方差折中；done 边界阻止优势跨 episode 传播。必须区分失败终止、回合超时、rollout 截断。

**3. 为什么优势归一化？在哪里做？**

减少不同批优势尺度变化对 Actor 更新的影响，稳定优化。当前 `rollout_storage.py:compute_returns()` 在整轮 rollout 上做零均值单位标准差，而非每个 minibatch 或 `ppo.update()` 里做；return 不归一化。均值中心化也可能改变单个样本优势符号，因此别说“完全不影响梯度方向的纯数学等价变换”。

**4. Critic 为什么可以部署丢弃？**

它只提供训练时基线/价值目标；策略前向已经学好 `obs → action`，无需实时估值。当前推理用 Actor 均值；若循环策略还需管理隐藏状态，但仍不要求 Critic。

**5. 并行数据怎么组织？为什么不全部回 CPU？**

每步 batch `[N,D]`，storage `[T,N,D]`，GAE 保留时间顺序，MLP update 再展平 `[T*N,D]` 并随机取 minibatch。GPU 仿真与训练可共享/就近访问张量，降低 CPU-Python/PCIe 开销；统计日志和部分初始化仍在 CPU。FPS 是所有环境步吞吐，不是视频帧率。

**6. reward 不涨怎么排查？**

按“配置与接口 → 仿真与有限值 → 任务可学性 → 奖励贡献 → 优化信号 → 泛化评估”顺序。先少量环境验证动作能驱动机器人、固定非零命令，再查摔倒率/误差、惩罚比例、实际 lr/噪声和 batch，最后做单变量修复与多种子复测。9.5 提供可直接使用的诊断表，不应凭 surrogate 正负就诊断发散。

**7. 仿真策略怎么上真机，gap 在哪里？**

gap 包括质量/惯量、接触/摩擦、执行器带宽与延迟、传感器噪声/偏置、速度估计、控制周期、关节顺序/方向、真实地形和安全限制。流程是标定/系统辨识 → 合理随机化与执行器建模 → 固定/压力工况评估 → 导出 Actor → 部署端严格复现预处理与动作接口 → 验证时延 → 仿真跨平台/硬件在环 → 受控低风险真机测试 → 扩大工况。JIT 只解决推理模型载入，不解决估计、控制与安全；本项目的 LSTM 执行器模型是仿真用，不等于必须在真实电机前再串一个同样网络。

**8. 为什么放历史、为什么归一化？**

历史动作/观测帮助处理执行器延迟、不可观测状态和步态相位；当前仅显式拼上次动作，并没有通用历史观测栈。LSTM 执行器在仿真内部有历史也不等于 Actor 自身是 RNN。`obs_scales` 是固定量纲缩放，`clip_observations` 防极值，**不是运行均值/方差标准化**。新增 history/RNN 需同步 reset、训练 storage 和部署状态，且历史不能泄露未来。

**9. 你调过最有效的参数是什么？**

这题没有可以替你编造的“标准最佳参数”。你目前可以诚实回答：“我已跑通基线并核对训练/控制链，正在做有固定指标的学习率、奖励比例与工况适配对照；尚未有足够实测证据判断最有效参数。”完成实验后按**症状 → 假设 → 精确改动 → 对齐预算与种子 → RMSE/摔倒率/成本结果 → 代价与边界**来讲。不要只说“调低 lr 效果好了”，也不要把本手册预期填成自己的数据。

**10. 你改过最有意思的东西是什么？**

未做挑战时同样不能说已实现。完成足端滑移挑战后，可讲为什么需要接触门控、怎样获得/刷新足端速度、负权重怎样绑定、怎样验证 reward 的零/非零案例，以及是否减少滑移同时保持跟踪能力。面试重点是需求、实现、验证、失败经验和取舍，不是奖励函数名字。

### 9.8 附录自测与项目外补强答案

附录 C 的算法题由 9.1–9.2 覆盖，工程题由 9.3–9.6 与第 10 节覆盖，曲线诊断由 9.5 覆盖；**“完成 10 组实验、写简历、精读论文”是交付任务，不能靠阅读参考答案视为完成**。

- **论文为何能分钟级学习步态？** 本项目对应 Rudin 等人的 *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*（arXiv:2109.11978）。核心是 GPU 端大规模并行仿真、减少数据传输、适合并行数据的 PPO 训练配置及地形课程，迁移还依赖随机化和执行器模型等。分钟级是论文特定硬件、任务、配置下的结果，不是任意机器人/本机训练时长保证；请实际精读实验设置再引用数字。
- **DQN 为什么需要 replay/target network？** Replay 复用样本、减弱连续数据相关性；target network 延缓 bootstrap 目标漂移。原始 DQN 用离散动作，12 维连续关节动作不适合直接枚举。
- **SAC/TD3 与 PPO 怎么选？** SAC 是带熵目标的 off-policy Actor-Critic；TD3 用双 Q、延迟策略更新和目标策略平滑改善确定性 Actor-Critic。都可复用 replay，通常适合样本昂贵的连续控制；PPO 数据利用率较低，但实现简单、并行仿真吞吐高时很实用。不是“PPO 一定更稳定”或“SAC 一定更快”，需在样本、算力与任务约束下测。
- **模仿学习/Offline RL/在线微调有何区别？** 行为克隆拟合专家动作，存在分布偏移；Offline RL 从固定数据学习价值/策略，需防分布外动作价值高估；在线微调从已有参数出发在目标环境采新数据再优化。本项目 PPO 是最后一种在线训练范式，不能把旧日志/旧 rollout 反复当作新 PPO 数据训练。

---

## 10. 工程调参与策略微调实战

### 10.1 先分清四种操作：你究竟在做什么？

| 操作 | 初始化 | 主要目的 | 注意点 |
|---|---|---|---|
| 从零训练 | 随机权重、全新优化器 | 建立任务基线 | 每组超参对照从同等起点开始 |
| 断点续训 resume | 通常恢复网络和优化器 | 继续同一目标的未完成训练 | 需记录计数、随机状态与配置，当前恢复不完整 |
| 策略微调 fine-tune | 从已有好策略加载权重，按需重置优化器 | 适配载荷/摩擦/速度范围/新奖励 | 每轮仍采目标环境的新数据，防遗忘 |
| 迁移到新机器人/观测 | 选择性迁移或重新训练 | 改输入、动作/动力学语义 | 维度匹配不等于语义匹配 |

这里“工程微调”不是只调网络 lr，更不是 LLM 的 LoRA。机器人 RL 的日常工作同时包括任务规格、数据/状态分布、控制接口、奖励、随机化、评估和安全。**你现在最缺的不是再背一个 PPO 公式，而是能把一个明确故障变成可复现、可验收的实验。**

### 10.2 从业务要求翻译为指标，而不是“reward 越高越好”

假设需求是“ANYmal 带额外载荷仍能跟踪命令，不能明显抖动或摔倒”。第一步先写评估协议：

- 场景：无载荷/目标载荷、前进/侧移/转弯/站立、指令切换、低摩擦及推搡；每个场景固定种子、时长、命令序列。
- 线速度 RMSE：$\sqrt{\operatorname{mean}((v_x-c_x)^2+(v_y-c_y)^2)}$，单位 m/s；yaw RMSE 单独统计 rad/s。
- 摔倒率：失败终止回合数 / 已结束回合数；timeout 另计。所有候选使用同样观察时长，不能把截短测试当成功。
- 抖动：动作差平方均值、关节加速度 RMS；滑移：仅接触阶段的足端水平速度均值/P95。
- 力矩安全：超过软限位/接近硬限位的比例。能耗若用 $\int\sum|\tau\dot q|dt$，需说明只是机械功耗 proxy，不等于电池耗电。
- 开销：训练环境交互数、墙钟时间、峰值显存、部署推理时延均值/P95/P99和最大值。

**验收条件在实验前定**：例如目标载荷下 RMSE 改善、摔倒率不恶化、无载荷指标在可接受回退范围内。具体阈值来自机器人/业务要求，不凭空规定所有项目都用同一个数。训练 reward 可帮助诊断，但改变奖励权重后不能直接拿它比较交付能力。

### 10.3 建立可信基线与公平调参顺序

1. **锁定版本和输入**：环境/依赖版本、训练与部署观测顺序、机器人资源、执行器模型、代码改动、种子、命令采样、实际生效配置都留档。工作区根目录当前不是 Git 仓库，不能假设有一个覆盖全部文件的 commit；可分别检查子项目，并保存文件快照/哈希。
2. **冒烟测试**：64 环境、3 轮只检查接口/shape/有限值/保存加载，不用于判断收敛。再做正常规模 baseline 与固定评估。
3. **确认任务可学、控制链正确**：机器人能立住、动作顺序正确，非零命令能产生运动；必要时先简化地形与命令。
4. **先奖励与观测，再优化器**：若不动/乱跳优先检查目标与约束，不要用 lr 掩盖接口或奖励错误。先测分项原始值和加权贡献。
5. **粗扫再细扫**：fixed lr 可先比较 `1e-4/3e-4/1e-3`；这些是搜索起点，不是保证好用的推荐答案。以单因素筛选后，再小规模测试明显的交互项。
6. **控制批量与预算**：`B=N×T`。1024 环境 24 步只有 24576 样本，4096 则是 98304；相同 iteration 不等样本量。性能比较至少按交互数对齐，工程速度还按时间比较。
7. **复测**：候选至少 3 个训练种子（如 1/2/3）；评估种子与训练分开，对所有候选复用同一测试集合，报告跨训练种子的均值、标准差及最差工况。4096 环境不是 4096 个训练种子。
8. **验证集选模型、测试集最终验收**：持续看同一压力场景再调参，相当于对测试集过拟合；保留未用于选择参数的工况。

快速命令（先激活已有 `legged_gym` 环境；日志路径是绝对路径）：

```bash
cd "/home/lixiang/AI Study/RL/legged_gym_study"
python "/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/scripts/train.py" \
  --task anymal_c_flat --headless --num_envs 64 --max_iterations 3 \
  --experiment_name engineering_smoke --run_name interface_check --seed 1
```

这条命令未修改任何超参。CLI **没有 `--learning_rate`、`--entropy_coef`**；这些参数要通过配置/自定义入口赋值。实验前确认改动真正作用在任务配置，而非被子类盖掉。

### 10.4 续训与微调最容易踩的 checkpoint 坑

源码：`/home/lixiang/AI Study/RL/rsl_rl/rsl_rl/runners/on_policy_runner.py` 的 `save()/load()/learn()`。

**当前 checkpoint 保存什么？** `model_state_dict`（Actor、Critic、可学习 std）、`optimizer_state_dict`、`iter`、`infos`。没有默认保存完整配置、Python/NumPy/Torch/CUDA RNG 状态、仿真状态、课程状态、rollout、累计交互数/训练时间、PPO 的独立 `learning_rate` 字段。

因此：

1. 默认 `--resume` 调 `runner.load()`，会恢复 Adam 动量和 param-group lr；**改配置 learning_rate 不保证下一步使用新值**。adaptive 又会用算法自身字段改写 optimizer lr，两者恢复后可能不一致，必须打印并明确设置两处。
2. 当前周期保存 `model_{it}.pt` 时，`self.current_learning_iteration` 未在循环中更新，只在 `learn()` 结束后增加；所以中间模型如 `model_250.pt` 的内部 `iter` 可能仍为 0。最终保存的计数才更新。文件名不能代替严格实验进度记录，生产化应修正保存计数语义并保存总步数。
3. `learn(num_learning_iterations=k)` 表示**从内部当前计数再运行 k 次**，CLI `--max_iterations 100` 在续训时是再跑 100 次，而不是“总量到 100 即停”。中间 checkpoint 的计数缺陷还会让日志横轴与实际进度错位。
4. `load(path, load_optimizer=False)` 只是不恢复优化器，仍恢复 `iter`；新微调实验通常随后把计数设 0，单独记源 checkpoint 和源训练预算。
5. 新奖励/工况下 Adam 旧动量、Critic 和探索 std 不一定合适。推荐先保留 Actor+Critic 权重、重置 Adam、用较小 fixed lr 起步；是否重置 Critic/增 std 必须另做对照。`init_noise_std` 对完整载入权重后的 std **不会自动重新生效**。
6. 必须校验输入维度、观测顺序/缩放、网络结构、动作顺序/缩放和控制语义；不匹配就拒绝加载或做明确迁移。不要用 `strict=False` 掩盖不理解的差异，也不要载入来历不明的 pickle checkpoint。
7. 指定源运行目录和编号；默认“最近运行”是按目录名排序，跨月份存在已知问题，新实验目录也可能被误选。**加载路径与输出目录分开**最稳，源模型保持只读。

工程上“精确恢复”还要明确是恢复统计训练进度，还是逐位复现轨迹。即便保存所有 RNG，GPU 物理求解也可能非确定，不能无条件承诺位级复现。

### 10.5 一个能落地的策略微调入口：载荷分布适配

**任务边界**：只针对默认 `anymal_c_flat` 的同结构前馈策略；加载可信的、已评估的 48→12 checkpoint。示例将训练附加质量分布设为 `[0,5]kg`，偏向正载荷，保留摩擦/噪声/推搡的默认设置。原 ANYmal 默认已有 `[-5,5]kg` 随机化，**这不是证明策略能泛化到从未见过的载荷**，而是目标工况重加权。该数值是仿真实验起点，不是你的真实机器人承载认证。

以下是完整示例入口，手动保存为 `/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/scripts/finetune_payload.py` 后可运行；**本次只补文档，没有创建或修改该训练脚本**。不需要新库，用当前 Isaac Gym、PyTorch 和标准库即可。

```python
import isaacgym  # 必须先于 torch 与 legged_gym 导入
import copy
import json
import os
from pathlib import Path

import torch
from legged_gym.envs import *  # 保持项目入口的任务注册模式
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, update_cfg_from_args


def main():
    args = get_args()
    if args.task != "anymal_c_flat" or args.resume:
        raise ValueError("本示例只接受 anymal_c_flat，勿加 --resume")
    source = os.environ.get("SOURCE_CHECKPOINT")
    if not source:
        raise ValueError("请将 SOURCE_CHECKPOINT 设为可信 checkpoint 的绝对路径")
    checkpoint = Path(source).expanduser()
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        raise ValueError("SOURCE_CHECKPOINT 必须是存在的绝对文件路径")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg, train_cfg = copy.deepcopy(env_cfg), copy.deepcopy(train_cfg)
    train_cfg.runner.resume = False
    train_cfg.runner.experiment_name = "payload_finetune"
    train_cfg.runner.max_iterations = 100
    train_cfg.algorithm.schedule = "fixed"
    train_cfg.algorithm.learning_rate = 1e-4
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0.0, 5.0]
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    env_cfg.seed = train_cfg.seed  # 外部配置入口需显式同步 CLI seed

    # 保存输入配置；环境内部使用的 reward_scales 会另乘 dt，需与原始权重区分。
    snapshot = {
        "env": class_to_dict(env_cfg),
        "train": class_to_dict(train_cfg),
        "args": vars(args),
        "source_checkpoint": str(checkpoint),
        "load_optimizer": False,
    }
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg
    )
    runner.load(str(checkpoint), load_optimizer=False)
    runner.current_learning_iteration = 0
    runner.alg.learning_rate = train_cfg.algorithm.learning_rate
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate

    with torch.no_grad():
        obs = env.get_observations().to(runner.device)
        actions = runner.alg.actor_critic.act_inference(obs)
        assert obs.shape == (env.num_envs, 48)
        assert actions.shape == (env.num_envs, 12)
        assert torch.isfinite(obs).all().item()
        assert torch.isfinite(actions).all().item()
        std = runner.alg.actor_critic.std
        assert torch.isfinite(std).all().item() and (std > 0).all().item()

    log_dir = Path(runner.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2)
    # 保留本次权重初始化，便于与更新后的候选作配对评估。
    runner.save(str(log_dir / "source_init.pt"))
    print("source:", checkpoint)
    print("output:", log_dir)
    print("lr:", runner.alg.learning_rate)
    print("mass range:", env.cfg.domain_rand.added_mass_range)
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
```

运行前先从已有日志选择**一个明确的可信 checkpoint**。下面用当前平地日志中修改时间最新的 `model_*.pt` 作为便捷候选，打印后仍需核对它是不是你要加载的好模型，不能把“最新”当成“最佳”：

```bash
cd "/home/lixiang/AI Study/RL/legged_gym_study"
export SOURCE_CHECKPOINT="$(python -c 'from pathlib import Path; root=Path("/home/lixiang/AI Study/RL/legged_gym_study/logs/flat_anymal_c"); paths=list(root.glob("*/model_*.pt")); assert paths, "没有可用 checkpoint，请先训练基线"; print(max(paths, key=lambda p:p.stat().st_mtime))')"
printf '将加载：%s\n' "$SOURCE_CHECKPOINT"
# 先冒烟，不验证性能
python "/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/scripts/finetune_payload.py" \
  --task anymal_c_flat --headless --num_envs 64 --max_iterations 3 \
  --run_name payload_smoke --seed 1
# 冒烟通过后才执行正式适配；100 次只是首轮预算，不保证收敛
python "/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/scripts/finetune_payload.py" \
  --task anymal_c_flat --headless --num_envs 1024 --max_iterations 100 \
  --run_name payload_s1 --seed 1
```

**为什么这样写？** 源路径不依赖日志根/最新 run，输出独立；深拷贝避免注册配置污染；加载权重不恢复旧 Adam；固定 lr 避免 adaptive 马上把“小学习率微调”改回去；同步 CLI seed 避免外部 config 的环境种子和训练配置不一致；计数清零明确是新实验；配置与初始化模型存档便于复现。仍需保存入口源码、依赖版本和源模型哈希，才能成为完整实验资产。

**怎样验证“微调有价值”？** 对照 A：源策略不更新；B：同结构从零训练目标分布；C：源策略微调。B/C 对齐目标任务新增交互数，另报告 C 已用的源训练成本，不能声称它端到端免费。若要比较“重置 Adam”的收益再加 D：恢复 Adam 的微调。用相同无载荷/固定 `+5kg`/摩擦/扰动评估验证 C 的目标改进和旧能力保留；示例代码仅演示训练入口，并未实现这套完整评测器。

### 10.6 评估不能直接等同于默认 play

`/home/lixiang/AI Study/RL/legged_gym_study/legged_gym/scripts/play.py` 是定性回放入口，会关闭观测噪声、摩擦随机化与 push，缩减地形，但不关闭质量随机化；它不是统一基准评估器。正式评估应该独立保存配置/指标并逐项控制：

1. **固定模型**：显式 load_run/checkpoint 或直接绝对路径；检查输出层/观测语义，使用确定性均值动作。
2. **固定质量工况**：关闭随机质量是恢复 nominal mass；要精确测试 `+5kg`，创建环境前用 `randomize_base_mass=True, added_mass_range=[5,5]`，让每个环境得到固定附加质量。`[0,0]` 测 nominal。
3. **固定摩擦工况**：随机化回调改变机器人 shape 的摩擦，不只地面。可用 `randomize_friction=True, friction_range=[mu,mu]` 固定 shape，地面 static/dynamic friction 也显式设置。平面配置注释指出平均组合，原 `[0,1.5]` 足底与地面 1 对应有效 `[0.5,1.25]`；只把足底改 0.1、地面保持 1，按该规则有效约为 0.55，仍可能在训练分布内。以实际仿真接触设置验证，不能仅改一个字段就声称 OOD。
4. **固定命令**：不要完全依赖随机重采样；在策略输入观测生成前设置命令并重新生成观测，保证控制目标与 Actor 看到的目标一致。reset 后的环境也要应用命令。站立、前进、侧移、转向和切换分别测试。
5. **指标对齐时刻**：步前命令与步后速度配对；失败原因和 terminal state 在自动 reset 前缓存，避免拿 reset 后速度当失败前速度。重置环境的统计与预热窗口应有明确规则。
6. **原始量统计**：分桶输出 RMSE、失败/超时、滑移、饱和率，不只累计 reward。可以用时间加权统计；各候选使用相同 reset/初始化、时长和种子。
7. **分级测试**：名义工况 → 训练分布内 → 边界工况 → 明确超出分布的压力测试。压力失败不自动代表模型不可用，但必须知道能力边界；不能拿不公平评测证明随机化有用。

现有 `Logger` 提供若干状态与奖励记录，但没有自动覆盖以上评估协议。面试中可说“我准备/实现了固定工况评估器”，只有实际实现并验证后才说后半句。

### 10.7 防灾难性遗忘、奖励黑客与“微调反而变差”

**遗忘**：目标载荷学好但原速度/摩擦能力退化。保留旧/新工况混合训练、渐进扩大新范围、适当降低 lr/更新 epochs；每隔固定预算评估旧能力并保留 Pareto 候选。不能简单往 PPO 塞旧 rollout 来解决遗忘；旧工况要由当前策略重新采样。

**奖励黑客**：新滑移惩罚让机器人不走；力矩惩罚让它长期低姿态或不追指令；只看最大回报误选怪步态。用独立原始指标验证，检查罚项门控、单位、累加方式和回合截断，做去除/加权消融。

**优化不适配**：新奖励让 Critic value loss 暴涨，可能是目标尺度/分布变了，不一定 Actor 已发散。比较保留/重置 Critic，检查 value clip 与奖励尺度，避免一次同时改 lr、熵、奖励和随机化。

**探索不适配**：已有 std 太小，新工况难探索；太大又破坏原有步态。先记录载入 std，适度增熵或显式设置 std 后另开对照，绝不盲目重置到 1.0。当前 std 直接是可学习参数，不是自动保证为正的 log-std 实现，需监测有限值与正值。

**约束旧策略**：可以增加相对源策略的 KL/动作蒸馏作为保留能力正则，但需修改训练代码、维护冻结源 Actor，并权衡适应能力。当前 PPO 的 `desired_kl` 比较的是本轮旧策略与更新策略，**不是永远相对最初的源策略**，不能混淆。

### 10.8 从 JIT 导出到可交付：必须补齐的部署工作

1. **输入契约**：明确 48 个观测的顺序、单位、坐标系、缩放/裁剪、四元数约定、关节映射、线速度估计来源、上一动作初始化规则。仿真真值不能直接视为硬件传感器输出。
2. **输出契约**：Actor 输出的是动作，不是最终绝对关节角；部署端复现动作裁剪、scale、默认姿态、底层接口与安全限幅。真实驱动模式必须与训练抽象相容。
3. **数值一致性测试**：在同一批代表性/边界观测上，比较 Python Actor 与 JIT 输出最大误差、shape、有限值；再检查 C++ 预处理与动作映射。测试不是只看模型是否成功 load。
4. **时序测试**：50Hz 策略周期不等于可占满 20ms；还需状态估计、通信和控制开销。测 warmup 后 P95/P99/最大时延、超时比例和内存，定义超时降级与 stale observation 处理。
5. **安全机制**：关节位置/速度/力矩硬限制、倾倒/异常检测、动作变化限幅、通信 watchdog、急停与安全站姿/阻尼回退；安全限制也应进入仿真和评估，否则部署额外 clip 会改变行为。
6. **分级验证**：离线输出一致性 → 仿真闭环/跨仿真器 → 硬件在环 → 专业人员监护、吊挂或安全区域内低速低载测试 → 扩大工况。不要直接在无保护真机上做随机探索。
7. **资产交付**：checkpoint、Actor 导出、配置与入口源码、机器人与执行器资源版本、依赖清单、评估报告、部署契约、已知失败场景、回滚模型与运行日志。导出文件固定名容易覆盖不同模型，另存版本化目录并记录源模型。

当前没有你的实际真机接口和测试记录，因此本节是交付清单，不能承诺“策略已可安全上真机”。

### 10.9 面试与实际工作中的工程追问

**“加载预训练策略后改 lr 就叫微调吗？”** 加载只是初始化，必须定义目标工况、决定 optimizer/Critic/std 如何恢复，用新分布采样，评估新能力与遗忘。当前 resume 会恢复优化器，所以还要确认实际 lr。

**“为什么微调用较小 lr，不冻结 Actor 大部分层？”** 已有步态不应被大步更新破坏，小 lr 是合理起点。但新动力学可能需要整体协同适应，机械冻结并非普适最优。若冻结，需明确冻结哪些参数及 optimizer 参数组、std/Critic 是否仍训练，并用对照验证。小 lr 也不保证不遗忘。

**“怎么证明微调比从零训练更好？”** 目标交互预算相同、多种子、同测试集合下比达到指标阈值的样本/时间与最终表现；源预训练成本单列，还要测旧能力。只比较两条 iteration 曲线不够。

**“多环境是不是等于样本独立、多 GPU 怎么扩展？”** 不完全独立，同一策略、初始分布和短期轨迹仍相关；独立 seed 是训练过程的独立复测。GPU 扩展需实际实现采样划分、梯度同步、吞吐与全局 batch 管理。当前有 `--horovod` 参数入口不代表已经实现/验证了完整分布式训练，不可凭参数名宣称经验。

**“改了观测维度，`strict=False` 能接着训吗？”** 同名层 shape mismatch 仍会报错；需过滤并明确迁移、重新初始化变化层，必要时蒸馏，然后验证输入语义。相同维度但顺序改变更危险，因为可能不报错却错误控制。

**“随机化越大越安全吗？”** 不是；过大可能学不动或过于保守，且期望回报不保证最坏情况安全。标定合理范围、逐渐增加难度，用边界/尾部工况与硬安全机制验收。

**“工程经验不足，面试怎么说？”** 可以真实表达：“我已跑通基线，理解采样/PPO/控制路径，也识别了配置覆盖和 checkpoint 恢复限制；目前在做固定载荷工况的策略适配，用 RMSE、摔倒率、遗忘和训练成本评估。”只有实验确实完成时才给改进数字。能讲清具体故障与验证方案，远好于虚构 10+ 组消融或真机经历。

### 10.10 给你当前阶段的最小练习与交付物

建议按可验收任务推进，天数只是参考，训练和排错时间依硬件而变：

| 顺序 | 亲手做的任务 | 完成标准 |
|---|---|---|
| 1 | 保存基线配置，核对 48→12、执行器路径、控制周期 | 能说明实际生效值和张量来源 |
| 2 | 显式加载一个 checkpoint 回放，核对内部 iter/optimizer lr | 能解释 resume 与权重初始化不同 |
| 3 | 做 fixed lr 单变量对照 | 样本预算对齐，有至少 3 种子候选复测 |
| 4 | 固定非零命令、质量、摩擦并输出 RMSE/失败率 | 原模型同配置复测一致到合理统计范围 |
| 5 | 保存并冒烟执行 10.5 入口，做目标载荷适配 | 新数据训练、独立日志、源模型不覆盖 |
| 6 | 源策略/从零/微调对照，加入无载荷回归测试 | 知道改进、退化、成本和不确定性 |
| 7 | 新增足端滑移奖励或接触观测二选一 | shape/有限值/物理零值测试通过，有固定评估 |
| 8 | 导出并比对 Actor/JIT，整理失败案例 | 输出一致性证据、部署契约、可回滚资产 |

实验记录建议创建在 `/home/lixiang/AI Study/RL/experiments.md`，每条至少包括：

- 实验 ID、日期、代码/依赖与资源版本、源 checkpoint/哈希；
- 假设、单个主要改动、最终配置快照；
- train seed / eval seeds、N/T/epochs/minibatch、交互与时间预算；
- 测试工况和各项原始指标的均值/方差/最差值；
- 训练曲线、典型失败视频/描述、相对基线收益与代价；
- 结论标为“支持/不支持/证据不足”，以及下一步最小验证。

**新增工程自测（必须实际做到再打勾）**：

- [ ] 能把任务要求写成固定场景、原始指标与验收阈值。
- [ ] 能分清从零训练、续训、微调、迁移，并核对实际 lr/std/optimizer 状态。
- [ ] 能在独立输出目录微调已有策略，保存配置、源模型与成本信息。
- [ ] 能公平比较源模型/从零/微调，多种子验证并测旧能力遗忘。
- [ ] 能识别默认 play 的配置覆盖，不把回放视频当完整评估。
- [ ] 能解释观测变更、关节顺序、控制周期、执行器模型和安全限幅的联动。
- [ ] 能验证导出推理的一致性，说明真机部署还缺哪些环节。

完成这些，你才有了可以具体讲述的工程调参和微调经验；不要求每次实验都成功，要求每个结论有证据、每次失败能定位并复现。
