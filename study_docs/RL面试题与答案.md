# 强化学习工程师面试题与答案
## —— 基于 legged_gym + rsl_rl + Isaac Gym 四足机器人项目

> 本文档基于工作区 `/home/lixiang/AI Study/RL` 的实际项目生成，覆盖该方向面试中最常被问到的问题。
> 项目技术栈：**Isaac Gym（GPU 并行物理仿真）+ legged_gym（腿式机器人环境）+ rsl_rl（PPO 算法库）**，
> 复现 ETH RSL 论文 *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*。
>
> 共 **10 大类、45+ 道题**，每题附参考答案与项目代码锚点。

---

## 目录

1. [项目介绍与架构类](#一项目介绍与架构类)
2. [强化学习基础理论](#二强化学习基础理论)
3. [PPO 算法深挖](#三ppo-算法深挖)
4. [Actor-Critic 网络与策略设计](#四actor-critic-网络与策略设计)
5. [MDP 设计：观测 / 动作 / 奖励 / 终止](#五mdp-设计观测--动作--奖励--终止)
6. [GPU 大规模并行仿真](#六gpu-大规模并行仿真)
7. [课程学习与域随机化](#七课程学习与域随机化)
8. [Sim-to-Real 迁移与部署](#八sim-to-real-迁移与部署)
9. [训练监控、调参与工程实践](#九训练监控调参与工程实践)
10. [编码与数学推导题](#十编码与数学推导题)

---

## 一、项目介绍与架构类

### Q1：请用 2 分钟介绍一下你做过的这个强化学习项目。

**参考答案（STAR 结构）：**

这是一个**四足机器人运动控制的深度强化学习项目**，基于 ETH Zurich RSL 实验室的开源框架，复现了"几分钟学会走路"的论文成果。

- **背景（S/T）**：传统四足机器人运动控制依赖人工设计控制器（如 MPC），开发周期长、对复杂地形适应性差。目标是用强化学习让 ANYmal C 四足机器人**端到端学习速度跟踪行走策略**，并能泛化到复杂地形。
- **行动（A）**：
  - 仿真底座用 NVIDIA Isaac Gym，**单卡并行 4096 个仿真环境**采集数据，把原本需要数天的训练压缩到几十分钟；
  - 算法层用 rsl_rl 库的 **PPO**（on-policy），Actor/Critic 为三层 MLP（平地任务 `[128,64,32]`），每次迭代采集 `4096 环境 × 24 步 ≈ 9.8 万样本`，做 5 epoch × 4 minibatch 更新；
  - 环境层（legged_gym）设计了 48 维本体感知观测、12 维关节位置偏移动作、10+ 项奖励塑形、地形课程学习与域随机化；
  - 动作经 PD 控制器（或 ANYmal 专用的**执行器神经网络**）转换为关节力矩，策略频率 50Hz、物理仿真 200Hz。
- **结果（R）**：策略能从随机初始化学会稳定跟踪 x/y/yaw 速度指令；训练好的 Actor 可通过 JIT 导出部署。

> 面试时可根据自己实际完成的实验补充具体数字（训练时长、收敛迭代数、速度跟踪误差等），**只讲自己真正跑过的实验**。

### Q2：这个项目的整体架构是怎样的？三个组件各承担什么角色？

**参考答案：**

```
legged_gym（环境） --观测obs/奖励reward--> rsl_rl（PPO算法） --动作action--> legged_gym（环境）
       |                                        ^
       +---------- isaacgym（GPU物理仿真底座）---+
```

| 组件 | 角色 | 关键文件 |
|---|---|---|
| `isaacgym` | GPU 物理仿真引擎，把刚体动力学、碰撞检测放在 GPU 上跑，状态直接是 GPU 张量 | `isaacgym/python/isaacgym/` |
| `legged_gym` | 环境层：定义 MDP 四要素（观测/动作/奖励/终止）、地形生成、课程学习、域随机化 | `legged_gym/envs/base/legged_robot.py` |
| `rsl_rl` | 算法层：PPO 实现、Actor-Critic 网络、Rollout 存储、训练循环，全部 GPU 张量运算 | `rsl_rl/rsl_rl/algorithms/ppo.py`、`runners/on_policy_runner.py` |

一次迭代的完整流程：

1. **Rollout**：`OnPolicyRunner.learn()` 驱动 —— Actor 采样动作 → 环境 step（内部 decimation=4 个物理步）→ 奖励/done → 写入 `RolloutStorage`；
2. **回报计算**：用 GAE 反向递推优势与回报，并做优势归一化；
3. **Update**：PPO 对这批数据做 5 epoch × 4 minibatch 的 clipped surrogate 更新，KL 自适应调学习率；
4. 记录 TensorBoard 日志、定期保存 checkpoint，清空 buffer 进入下一轮。

### Q3：配置系统是怎么组织的？如果让你改一个奖励权重，你怎么确认它生效了？

**参考答案：**

配置采用**嵌套类 + 多层继承**的方式（不是 dataclass）：

- 继承链（以平地任务为例）：
  - 环境：`AnymalCFlatCfg → AnymalCRoughCfg → LeggedRobotCfg → BaseConfig`
  - 算法：`AnymalCFlatCfgPPO → AnymalCRoughCfgPPO → LeggedRobotCfgPPO`
- `BaseConfig.__init__()` 调 `init_member_classes()` 递归把嵌套类实例化，之后用属性链访问（`cfg.rewards.scales.tracking_lin_vel`）。
- 任务注册表（`legged_gym/envs/__init__.py` 的 `task_registry`）把任务名映射到 (环境类, 环境配置, 训练配置)。

**确认改动生效的方法**：不能只看配置文件——子类可能覆盖父类，CLI 参数也可能再覆盖。正确做法是在**创建环境/runner 之前打印实际生效的配置对象**（如 `print(env_cfg.rewards.scales.xxx)`）。另外注意：零权重的奖励项会在 `_prepare_reward_function()` 中被移除，写了函数不等于启用。

---

## 二、强化学习基础理论

### Q4：什么是 MDP？本项目的 MDP 五要素分别是什么？

**参考答案：**

MDP（马尔可夫决策过程）= $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$：

| 要素 | 本项目实例 |
|---|---|
| 状态 $\mathcal{S}$ | 机器人完整物理状态（各刚体位置/速度等），仿真内部维护 |
| 观测 $\mathcal{O}$（POMDP 视角） | 48 维向量：基座线速度(3) + 角速度(3) + 重力投影(3) + 速度指令(3) + 关节位置(12) + 关节速度(12) + 上一步动作(12)；rough 任务再加 187 维地形高度 = 235 维 |
| 动作 $\mathcal{A}$ | 12 维连续向量，关节目标位置相对默认站姿的偏移 |
| 转移 $P$ | PhysX 物理引擎决定（动力学 + 接触） |
| 奖励 $R$ | 速度跟踪指数奖励 + 姿态/力矩/平滑性惩罚等加权和 |
| 折扣 $\gamma$ | PPO 配置中 `gamma=0.99` |

严格说这是一个 **POMDP**：策略拿不到完整状态（如地面摩擦系数、接触力、真机上的绝对位置），这也是要用域随机化/历史帧等手段补偿的原因之一。

### Q5：on-policy 和 off-policy 的区别？PPO 属于哪种，这带来了什么工程后果？

**参考答案：**

- **On-policy**：训练数据必须由**当前策略**采集，用完即弃。代表：PPO、TRPO、A2C。
- **Off-policy**：可以复用旧策略/他人策略存进 replay buffer 的数据。代表：DQN、SAC、TD3。

PPO 是 on-policy，本项目代码里体现为 `rsl_rl/rsl_rl/algorithms/ppo.py` 中 `update()` 末尾的 `self.storage.clear()`——每次更新完清空缓冲区。

**工程后果**：PPO 样本效率低（每条数据平均只被用 5 个 epoch 就丢弃），所以**采样吞吐成为瓶颈**——这正是本项目要用 Isaac Gym 在单卡上并行 4096 个环境的核心动机。Off-policy 算法靠 replay buffer 复用数据，对并行仿真的需求没有这么极端。

### Q6：写出策略梯度定理，并解释为什么引入基线（baseline）和优势函数。

**参考答案：**

策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, Q^{\pi}(s_t, a_t)\right]$$

直接用回报 $Q$ 的问题是**方差大**：同一状态下所有动作的回报都含一个公共项 $V(s)$，梯度估计噪声大。

引入基线 $b(s)$ 不改变梯度的期望（因为 $\mathbb{E}_{a\sim\pi}[\nabla\log\pi(a|s)\,b(s)] = 0$），但能显著降方差。最自然的基线是状态价值 $V(s)$，于是得到**优势函数**：

$$A(s,a) = Q(s,a) - V(s)$$

直觉：只奖励"比平均水平好"的动作、惩罚"比平均差"的动作。本项目用 GAE 估计 $A$（见 Q12），并在 `compute_returns()` 里对整轮 rollout 做 $(A-\bar A)/(\sigma_A+10^{-8})$ 归一化，进一步稳定更新尺度。

### Q7：折扣因子 γ 的作用？γ=0.99 意味着什么？

**参考答案：**

γ 控制未来奖励的重要程度：回报 $G_t=\sum_k \gamma^k r_{t+k}$。

- **数学上**：γ<1 保证无限时域下回报有界，值函数收敛；
- **算法上**：γ 越小，估计方差越小、偏差越大（更"近视"）；γ 越接近 1，目光越远但估计更难；
- **本项目**：`gamma=0.99`，策略周期 0.02s，有效视野约 $\frac{1}{1-\gamma}=100$ 步 ≈ 2 秒——足够覆盖一个完整步态周期，对行走任务合理。γ 同时改变任务定义本身（智能体优化的是折扣回报），不只是数值技巧。

---


## 三、PPO 算法深挖

### Q8：写出 PPO-Clip 的目标函数，并解释 clip 的"保守性"体现在哪里。

**参考答案：**

定义概率比 $r_t(\theta)=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，PPO-Clip 目标：

$$L^{CLIP}(\theta)=\mathbb{E}_t\left[\min\Big(r_t(\theta)\hat A_t,\ \text{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big)\hat A_t\Big)\right]$$

代码实现（`ppo.py`，注意取了相反数变最小化）：

```python
ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
surrogate = -torch.squeeze(advantages_batch) * ratio
surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)   # clip_param = 0.2
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

**保守性来自对"有利方向"的收益截断**：
- $A>0$（好动作）：ratio 超过 $1+\epsilon$ 后目标不再增长 → 不再奖励继续增大该动作概率；
- $A<0$（坏动作）：ratio 低于 $1-\epsilon$ 后目标不再增长 → 不再奖励继续压低概率。

注意三个常见误区：clip **不直接裁剪动作**、**不限制参数更新幅度**、也**不能严格保证 KL 不超标**（所以本项目还要叠加 KL 自适应学习率，见 Q11）。

### Q9：为什么 ratio 要用 `exp(log_prob_new − log_prob_old)` 计算，而不是直接算两个概率相除？

**参考答案：**

本项目动作是 12 维对角高斯策略，联合概率密度是 12 个维度密度的**乘积**，数值上可能极小，直接计算分子、分母再相除容易**下溢**。

先对各维 log-density 求和（`get_actions_log_prob()` 里 `.sum(dim=-1)`），再用 `exp(logp_new − logp_old)` 得到同一比值，数值稳定得多，且 log 形式天然方便反向传播。

另外强调：连续动作空间下这是**概率密度比**，不是"选中该动作的概率"（连续分布单点概率为 0）。密度可以大于 1。

### Q10：为什么在 PPO 更新阶段，`old_actions_log_prob` 必须保持不变？

**参考答案：**

`old_actions_log_prob` 是**行为策略**（采样这批数据时的策略）产生这些动作的 log 概率，是 ratio 的**分母基准**。PPO 的样本复用能力全部来自"新策略相对旧策略改变了多少"这个比值。

如果每个 epoch 都用当前策略重算分母，ratio 会恒等于 1，clip 完全失效，算法退化成普通策略梯度且失去样本复用的意义。

同样被冻结的还有：采样动作本身、旧价值 $V_{old}$（value clip 用）、旧策略的 μ/σ（算 KL 用）、以及本轮的 GAE 目标。更新的只有**新策略对同一批动作的 log 概率**。

### Q11：本项目如何根据 KL 散度自适应调整学习率？为什么选 KL 作为信号？

**参考答案：**

`ppo.py` 的 `update()` 中（`schedule='adaptive'`，`desired_kl=0.01`）：

1. 用对角高斯的**解析 KL 公式**计算 $D_{KL}(\pi_{old}\| \pi_{new})$ 的 batch 均值：

$$D_{KL}=\sum_i\left[\log\frac{\sigma_{new}}{\sigma_{old}}+\frac{\sigma_{old}^2+(\mu_{old}-\mu_{new})^2}{2\sigma_{new}^2}-\frac12\right]$$

2. 规则：`kl > 2×desired_kl` → 学习率 ÷1.5（下限 1e-5）；`kl < desired_kl/2` → 学习率 ×1.5（上限 1e-2）；写回 Adam 所有参数组。

**为什么用 KL**：KL 度量的是**行为分布**层面的变化（"新策略在这批状态上的动作分布偏了多少"），比参数空间的距离更贴近我们真正关心的"策略别走太远"。

注意边界：这是**启发式调速**，不是 TRPO 的硬约束，也不是 KL early stopping（更新不会中途停止，只是下一轮 lr 变小）；KL 大说明更新偏猛，**不等于**任务性能已经下降。

### Q12：详解 GAE。λ 和 γ 各自权衡什么？λ=0 / λ=1 两个极端是什么？

**参考答案：**

GAE（Generalized Advantage Estimation）是多步 TD 误差的指数加权平均：

$$\delta_t = r_t + \gamma(1-d_t)V(s_{t+1}) - V(s_t)$$
$$\hat A_t = \delta_t + \gamma\lambda(1-d_t)\hat A_{t+1}, \qquad R_t = \hat A_t + V(s_t)$$

代码在 `rsl_rl/rsl_rl/storage/rollout_storage.py` 的 `compute_returns()`，从最后一步**反向递推**（所以必须先算 GAE 再打乱数据，不能先 shuffle 轨迹）。

- **λ（本项目 0.95）**：偏差-方差权衡。
  - λ=0：退化为单步 TD，$A_t=\delta_t$，完全依赖 Critic → 低方差、高偏差；
  - λ=1：接近 Monte Carlo 长程回报 → 低偏差、高方差；
  - 0.95 是常用折中，不是普适最优。
- **γ（0.99）**：既定义任务（优化折扣回报）又影响估计难度。

另外注意终止处理：**真正失败终止（done=1）不 bootstrap**（$(1-d_t)$ 置零）；**时间截断（timeout）应该 bootstrap**——本项目通过 `send_timeouts=True` 在 `process_env_step()` 中给 timeout 步的奖励补上当前状态的价值估计。

### Q13：PPO 的完整损失函数由哪几项组成？本项目各系数是多少？

**参考答案：**

```python
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
```

| 项 | 作用 | 本项目配置 |
|---|---|---|
| Clipped surrogate loss | 策略改进主项 | `clip_param=0.2` |
| 价值损失（MSE 或 clipped MSE） | Critic 拟合 GAE 回报 | `value_loss_coef=1.0`，`use_clipped_value_loss=True` |
| 负熵项 | 鼓励探索，防止策略过早收敛 | `entropy_coef=0.01` |

更新流程固定为：`zero_grad() → backward() → clip_grad_norm_(max_grad_norm=1.0) → optimizer.step()`。

### Q14：为什么要对 value loss 也做 clip？

**参考答案：**

PPO 对同一批数据做多个 epoch 更新，Critic 可能在少数几步内把价值估计改得面目全非，导致优势目标剧烈漂移、训练不稳。Value clipping：

$$V_{clip} = V_{old} + \text{clip}(V_\theta - V_{old}, -\epsilon, \epsilon)$$
$$L_V = \mathbb{E}\left[\max\big((V_\theta - R)^2,\ (V_{clip} - R)^2\big)\right]$$

取两者较大误差 = 保守更新。

两个澄清：① 裁剪的是**相对旧价值的变化量**，不是把价值限制在 [-0.2, 0.2]；② 本项目复用了 `clip_param=0.2`，其数值意义受奖励尺度影响，若大改奖励函数需要重新审视这个值。

### Q15：PPO 与 TRPO、A2C、SAC、DQN 的关键区别？为什么机器人控制常选 PPO？

**参考答案：**

| 算法 | 类型 | 核心机制 | 局限 |
|---|---|---|---|
| A2C | on-policy | 优势函数 + 同步多环境，无信赖域 | 步长敏感，易崩 |
| TRPO | on-policy | KL 硬约束 + 共轭梯度 + 线搜索 | 二阶方法，实现复杂、计算贵 |
| **PPO** | on-policy | clip 近似信赖域，一阶方法 | 样本效率低 |
| SAC | off-policy | 最大熵 + 双 Q | 连续控制强，但 replay buffer 对高维机器人状态实现重 |
| DQN | off-policy | Q 学习 + target network | 只适合离散动作 |

选 PPO 的原因：**简单稳定、对超参相对鲁棒、与大规模并行采样天然契合**（on-policy 的最大短板——数据不可复用——正好被 4096 并行环境的高吞吐补上）。机器人控制要求训练稳定可复现，PPO 的一阶实现远比 TRPO 易维护。

---


## 四、Actor-Critic 网络与策略设计

### Q16：本项目的 Actor-Critic 网络结构是怎样的？动作随机性从哪来？

**参考答案：**

见 `rsl_rl/rsl_rl/modules/actor_critic.py`：

- **Actor**：MLP，输入 48 维观测，隐藏层 `[128, 64, 32]`（平地 ANYmal 配置）+ ELU，输出 12 维高斯策略**均值 μ**；
- **Critic**：同构 MLP，输出标量 V(s)；
- **随机性**：标准差 σ **不是网络输出**，而是一个与状态无关的**可学习参数** `self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))`，初始 1.0；
- 每次 `act()` 先 `update_distribution()` 用当前观测算 μ，再构建 `Normal(μ, σ)` 采样。代码里 `Normal(mean, mean*0. + self.std)` 的写法把 std 广播到 batch 维且不引入对 mean 的梯度。

这种"state-independent std"设计简单、稳定，代价是探索噪声不能随状态自适应。注意这里的 std 直接是可学习参数（非 log-std 参数化），不自动保证为正，训练中要监控 `Policy/mean_noise_std`。

### Q17：训练时 `act()` 和部署时 `act_inference()` 有什么区别？为什么部署只需要 Actor？

**参考答案：**

- `act()`：从 `Normal(μ, σ)` **随机采样**动作——训练时需要探索；
- `act_inference()`：直接输出**均值 μ**（确定性策略）——部署/评估时用，性能更好且行为可复现。

部署只需 Actor 的原因：Critic 的唯一作用是训练时构造优势函数和价值目标（GAE），决策链路 `obs → action` 不经过它。所以 `play.py` 导出 JIT 策略时只导出 Actor 部分——导出模型**不包含** Critic、观测预处理、PD 控制、执行器网络和安全限幅，这些要在部署端另行实现。

### Q18：什么是特权观测（privileged observation）/ 非对称 Actor-Critic？本项目支持吗？

**参考答案：**

非对称 Actor-Critic：训练时 **Critic 可以看到仿真中的"真值"信息**（如地面摩擦系数、接触力、精确地形高度），Actor 只能用本体可得的观测。Critic 估得更准 → 优势信号更好 → 但部署时只需要 Actor，所以特权信息"只在训练期作弊"。

本项目代码已留好接口：`OnPolicyRunner.__init__()` 中若 `env.num_privileged_obs is not None`，Critic 就用特权观测维度；`RolloutStorage` 也有专门的 `privileged_observations` 缓冲。默认配置 `num_privileged_obs=None`（未启用），但这是近乎改一行配置即可开启的、被 RSL 后续工作广泛验证的 sim-to-real 技巧。

### Q19：为什么观测要放在机体坐标系表示，而不是世界坐标系？

**参考答案：**

本项目 48 维观测中，线速度、角速度、重力向量都投影在**机体系**：

1. **泛化性**：策略学到的是"身体相对重力和指令该怎么动"，与世界绝对位置/朝向解耦——同一策略在场地的任何位置、朝任何方向都适用，避免策略记忆地图；
2. **平稳性**：观测分布不随机器人跑远而漂移，值函数更好拟合；
3. 代价：真机上基座线速度没有直接传感器，需要状态估计器（如里程计/融合 IMU），仿真真值不能直接搬到硬件。

---


## 五、MDP 设计：观测 / 动作 / 奖励 / 终止

### Q20：48 维观测具体由什么组成？每一维的物理意义？

**参考答案：**

| 切片 | 内容 | 维度 |
|---|---|---|
| `[0:3]` | 机体系基座线速度 × scale | 3 |
| `[3:6]` | 机体系基座角速度 × scale | 3 |
| `[6:9]` | 重力单位向量在机体系的投影（隐含姿态） | 3 |
| `[9:12]` | 速度指令 (vx, vy, wyaw) × scales | 3 |
| `[12:24]` | 关节位置 − 默认站姿角 × scale | 12 |
| `[24:36]` | 关节速度 × scale | 12 |
| `[36:48]` | 上一步执行的动作（历史动作） | 12 |

rough 任务额外拼接 17×11=187 个机身周围地形高度采样点 → 235 维。

设计亮点：① **加入上一步动作**让策略能感知动作时序、平滑输出；② **重力投影**替代欧拉角，无奇异性；③ 各项都有 scale 归一，避免不同量纲碾压网络。

### Q21：为什么动作设计成"关节位置偏移"而不是直接输出力矩？

**参考答案：**

动作定义：$q_{des} = q_{default} + s_a \cdot a$，$s_a$=`action_scale=0.5`。

- **直接力矩控制**：策略要从零学"哪个关节该出多大力"，探索空间巨大，初期动作近乎随机抖动，极易摔且难收敛；
- **位置目标控制**：底层 PD 控制器 $\tau = K_p(q_{des}-q) - K_d\dot q$ 已提供稳定性（类似生物的脊髓反射），策略只需在默认站姿附近学偏移，搜索空间小、天然平滑、易收敛。

代价是性能上限依赖执行器模型精度。本项目两条路径：A1 等用理想 PD；ANYmal 系列用**预训练执行器网络**（LSTM，建模 ANYDrive SEA 电机的真实响应）替代理想 PD，减小 sim-to-real 的执行器误差。

### Q22：策略频率和物理仿真频率分别是多少？为什么这样设计？

**参考答案：**

- 物理仿真步长 `dt=0.005s`（200Hz），`decimation=4` → **策略周期 0.02s（50Hz）**；
- 同一个动作保持 4 个物理步，每个物理步重新计算力矩（PD 在 200Hz 跑），不是每 0.005s 重新推理一次策略。

原因：① 物理步要足够小才能稳定求解接触动力学；② 50Hz 与真实机器人控制频率对齐（通信、状态估计都有延迟），在仿真里用更高策略频率反而学到真机上不可实现的策略；③ 降低策略推理次数也省算力。

### Q23：奖励函数是怎么设计的？列举主要奖励项及其作用，谈谈奖励塑形的原则。

**参考答案：**

总奖励 = 各加权项之和 × 策略 dt，然后 `only_positive_rewards=True` 把总和下限截为 0。平地任务有效项：

| 奖励项 | 权重 | 作用 |
|---|---|---|
| `tracking_lin_vel` | +1.0 | **主任务**：$\exp(-\|v_{xy}-v_{cmd}\|^2/\sigma)$ 指数型速度跟踪 |
| `tracking_ang_vel` | +0.5 | yaw 角速度跟踪 |
| `lin_vel_z` | −2.0 | 抑制上下跳动 |
| `ang_vel_xy` | −0.05 | 保持机身平稳 |
| `orientation` | −5.0 | 惩罚倾斜（重力投影 xy 分量平方） |
| `torques` | −2.5e-5 | 力矩平方惩罚（节能，**不是真实电耗**） |
| `dof_acc` | −2.5e-7 | 关节加速度惩罚（平滑） |
| `action_rate` | −0.01 | 相邻动作差惩罚（平滑、防抖） |
| `collision` | −1.0 | 非足端部位碰撞惩罚 |
| `feet_air_time` | +2.0 | 鼓励腾空时间（塑造清晰步态，非零指令门控） |

**塑形原则**：
1. **任务项用指数奖励**（有界、平滑、梯度自然衰减），惩罚项用平方；
2. 权重体现优先级：任务 >> 姿态 > 平滑/能耗；
3. 每项函数量纲不同（指数 vs 平方），**不能只看权重绝对值判断重要性**；
4. 小心奖励黑客（见 Q38）；任何惩罚都要检查是否误伤正常行为（门控条件）；
5. 零权重项会被移除，写了 `_reward_xxx` 函数不等于启用。

### Q24：终止条件怎么设计？为什么要区分"失败终止"和"超时终止"？

**参考答案：**

`check_termination()`：基座等指定部位接触力 >1N（摔倒）**或** `episode_length > max_episode_length`（20s）→ reset。

**必须区分的原因在于价值 bootstrap**：
- 失败终止：真实到达终止状态，$V(s_{terminal})=0$，TD 目标不 bootstrap；
- 超时终止：任务本身没完，是人为截断，**应该用 $V(s_{t+1})$ 自举**，否则 Critic 会学到"活到 20 秒边缘的状态价值为 0"，策略在回合末尾行为退化。

本项目通过 `send_timeouts=True` 把 `time_outs` 传给算法，在 `process_env_step()` 中对 timeout 步用当前价值补偿。工程上两者还要分开统计——timeout 不是摔倒，混为一谈会误判策略水平。

---


## 六、GPU 大规模并行仿真

### Q25：为什么 4096 个并行环境能让训练快这么多？加速比是 4096 倍吗？

**参考答案：**

**为什么快**：
1. Isaac Gym 把刚体动力学、碰撞求解全部放在 GPU 上，4096 个机器人的物理状态是一次性批量推进的；
2. 仿真状态直接就是 **GPU 张量**，策略前向、环境 step、奖励计算全在 GPU，**避免了传统方案每个 step 在 CPU↔GPU 之间搬运整批状态**（PCIe 瓶颈）；
3. Actor 以 batch=4096 推理，GPU 利用率拉满。

**为什么 PPO 特别受益**：on-policy 算法必须反复用当前策略采新数据，没有 replay buffer 可以复用，采样吞吐直接决定训练速度。

**不是 4096 倍**：碰撞求解的串行部分、显存带宽、kernel launch、同步与日志都有开销，且 4096 个环境共享同一个策略（不是 4096 个独立训练进程）。实际意义是把"几天"压到"几十分钟"量级。

### Q26：多环境采的数据是独立的吗？样本间的相关性怎么处理？

**参考答案：**

不完全独立：4096 个环境共享同一策略、同一初始分布，同一时刻的轨迹仍有相关性。

本项目的去相关手段：
1. 环境间**随机初始状态**、随机地形、域随机化参数各异；
2. `init_at_random_ep_len=True`：训练开始时随机化各环境已进行的 episode 长度，**错开 reset 时机**，避免所有环境同步终止/重生造成的周期性相关；
3. 更新前把 [T, N] 数据**展平打乱**再切 minibatch；
4. 指令周期性重采样（flat 任务 4s / rough 10s）。

### Q27：一次 PPO 迭代的完整数据流和张量形状是怎样的？

**参考答案（平地默认 N=4096, T=24）**：

```
obs [4096, 48] → Actor采样 actions [4096, 12] → env.step
   ↓ 写入 RolloutStorage（时间维在前）
storage: observations [24, 4096, 48], actions [24, 4096, 12],
         rewards/values/log_prob [24, 4096, 1], mu/sigma [24, 4096, 12]
   ↓ 采满 24 步
用最后的 critic_obs 算 V 边界 → 反向递推 GAE → returns/advantages
   ↓ 展平
总样本 B = 24×4096 = 98304，优势归一化
   ↓ 5 epoch × 4 minibatch = 20 次梯度更新，每批 24576 样本
   ↓
storage.clear()，进入下一轮
```

关键约束：**必须先算完 GAE 再打乱**——GAE 依赖时间顺序，shuffle 只能在维度展平之后做。

---

## 七、课程学习与域随机化

### Q28：本项目的地形课程学习（terrain curriculum）是怎么实现的？

**参考答案：**

- 地形是一张 `num_rows=10`（难度等级）× `num_cols=20`（地形类型变体）的网格，类型包括平滑斜坡/粗糙斜坡/上楼梯/下楼梯/离散台阶（比例 `[0.1,0.1,0.35,0.25,0.2]`）；
- 每个环境被分配到某一行（难度），机器人走得足够远 → **升级**到更难一行；走得太近 → **降级**；
- `max_init_terrain_level=5` 控制初始最高难度，防止一上来就把机器人扔进高难度地形。

本质：**自动调节任务难度使其始终处于"学习区"**——太简单没有学习信号，太难拿不到奖励。这与 PPO 的 KL 调速一个思想，只是作用在任务维度。

### Q29：本项目用了哪些域随机化？各自的动机是什么？

**参考答案（`cfg.domain_rand`）**：

| 随机化项 | 范围 | 动机 |
|---|---|---|
| 地面摩擦 `randomize_friction` | `[0.5, 1.25]`（flat 任务 `[0, 1.5]`） | 真实地面摩擦未知且多变 |
| 基座附加质量 `randomize_base_mass` | `[−1, 1]` kg | 模拟电池/负载差异、建模误差 |
| 随机推搡 `push_robots` | 每 15s 一次，xy 速度 ≤1 m/s | 扰动鲁棒性、跌倒恢复 |
| 观测噪声 `noise.add_noise` | 各观测分量独立 `noise_scales` | 模拟传感器噪声 |
| （可选）执行器网络 | — | 建模真实电机响应，弥补理想执行器差距 |

动机一句话：**与其把仿真建得无限精确，不如让策略对一片参数分布都鲁棒**——真实系统只要落在分布内就能工作。

### Q30：域随机化是不是范围越大越好？

**参考答案：**

不是。三个理由：

1. **可学性**：范围过大（如摩擦接近 0 的镜面地面）任务本身不可解，策略学不动；
2. **保守性**：策略要在整个分布上都表现好，会牺牲标称性能（"平均好、处处平庸"）；
3. **期望 ≠ 最坏情况**：期望回报优化不保证边界工况安全。

正确姿势：先用先验/系统辨识**标定合理范围**，训练中**渐进扩大**（与课程学习结合），最后用**边界工况和分布外压力测试**验收，并配合硬安全机制（见 Q33）。

---

## 八、Sim-to-Real 迁移与部署

### Q31：sim-to-real 的差距（reality gap）主要来自哪里？本项目用了哪些手段缩小它？

**参考答案：**

**差距来源**：
1. **动力学误差**：执行器响应（理想 PD vs 真实电机的带宽/饱和/迟滞）、接触模型、连杆惯量误差；
2. **感知误差**：仿真给真值，真机只有带噪声/延迟的传感器估计；
3. **环境差异**：地面摩擦、地形几何。

**本项目的手段**：
- **执行器网络**（ANYmal）：用真实电机数据预训练 LSTM 模拟 ANYDrive 响应，替代理想 PD——这是针对性最强的一招；
- **域随机化**：摩擦/质量/推力/观测噪声（见 Q29）；
- **机体系观测 + 历史动作**：不依赖世界坐标绝对量；
- **50Hz 策略频率**：与真机控制周期对齐；
- **动作裁剪 + action_rate 惩罚**：输出平滑有界，避免真机执行不了的动作。

### Q32：从训练好的 checkpoint 到真机部署，完整的交付流程有哪些环节？

**参考答案：**

1. **导出**：`play.py` 里 JIT trace 导出 Actor（仅前馈策略，**不含**观测预处理、PD、执行器网络、安全层）；
2. **输入契约**：明确 48 维观测的顺序、单位、坐标系、缩放、关节映射、线速度估计来源——仿真真值不能直接当传感器输出；
3. **输出契约**：部署端复现动作 clip、action_scale、默认站姿叠加、PD/执行器接口与安全限幅；
4. **数值一致性测试**：同一批代表性/边界观测上比对 Python Actor 与 JIT/C++ 端输出的最大误差（不是只测"能 load"）；
5. **时序测试**：50Hz 周期≠能占满 20ms，要测 P95/P99 延迟，定义超时降级策略；
6. **安全机制**：关节位置/速度/力矩硬限幅、倾倒检测、watchdog、急停——且这些限制**应该进仿真训练**，否则部署端的 clip 会改变策略行为；
7. **分级验证**：离线一致性 → 仿真闭环 → 硬件在环 → 吊挂/安全区低速测试 → 逐步扩大工况；
8. **资产交付**：版本化的 checkpoint、配置、评估报告、已知失败场景、回滚模型。

### Q33：为什么部署时只导出 Actor 还不够，还要在部署端重新实现 PD 控制和安全限幅？

**参考答案：**

JIT 导出的只是 `obs → action` 的神经网络前向。而真实控制链路是：

```
传感器 → 状态估计 → 观测组装(48维, 含scale) → Actor → 动作a
      → clip → ×action_scale → +默认站姿 → q_des → PD/执行器 → 力矩τ → 电机
```

中间每一环都是训练时策略"见过"的分布的一部分：如果部署端漏掉 clip 或 scale，Actor 收到的闭环反馈就偏离训练分布，行为退化甚至危险。安全限幅同理——它是策略行为边界的组成部分，必须在仿真训练时就存在，而不是部署时才加上。

---


## 九、训练监控、调参与工程实践

### Q34：训练时你主要看哪些 TensorBoard 曲线？各自的健康状态是什么？

**参考答案：**

| 曲线 | 含义 | 健康状态 |
|---|---|---|
| `Train/mean_reward` | 近 100 回合平均累计奖励 | 长期上升后进入平台期 |
| `Train/mean_episode_length` | 平均存活步数 | 变长（少摔）；到达上限后失去区分度 |
| `Episode/rew_*` | 各奖励项的回合累计 | 任务项上升，惩罚项趋近 0 |
| `Loss/value_function` | Critic 拟合误差 | 不发散、无 NaN（不要求单调下降——目标是移动的） |
| `Loss/surrogate` | PPO 策略损失 | 允许正负波动，**不能仅凭高低判断好坏** |
| `Policy/mean_noise_std` | 探索噪声 σ | 缓慢下降或持平；**过早归零=探索崩塌** |
| `Train/fps` 等 | 吞吐 | 异常下降提示性能/死锁问题 |

### Q35：如果训练 reward 一直不涨，你的排查顺序是什么？

**参考答案（系统性排查清单）：**

1. **任务可行性**：奖励尺度是否合理？终止条件是否太苛刻（秒摔→没有学习信号）？初始难度是否过高（课程起点）？
2. **观测/动作链路**：观测维度/顺序/缩放是否正确？动作到力矩的链路（clip、scale、PD 增益）是否生效？——用调试器在 `train.py:60`、`on_policy_runner.py` 采样循环下断点看实际张量；
3. **奖励定义**：打印各 `rew_*` 分项，看是任务项不涨还是某惩罚项爆炸（奖励黑客）；
4. **优化层面**：value loss 是否 NaN/爆炸？std 是否已崩塌（探索消失）？KL 是否长期顶格（lr 被压到 1e-5）？
5. **配置生效性**：改的参数是否真的生效了（继承链覆盖问题，见 Q3）？
6. **随机性**：换 seed 复测，排除单次坏运气。

原则：**一次只改一个变量**，样本预算对齐，多种子验证。

### Q36：PPO 最敏感的超参数有哪些？调错各自有什么症状？

**参考答案：**

| 超参 | 本项目值 | 调错症状 |
|---|---|---|
| 学习率 | adaptive, 初始 1e-3 | 过大：KL 顶格、reward 震荡崩坏；过小：收敛极慢 |
| `clip_param` ε | 0.2 | 过大≈无约束策略梯度，不稳；过小：学习保守缓慢 |
| `gamma` | 0.99 | 过小：近视、步态短视；过大：值函数难拟合、方差大 |
| `lam` (GAE) | 0.95 | 接近 0：偏差高；接近 1：方差高 |
| `entropy_coef` | 0.01 | 过大：收敛慢、动作乱；过小：std 早塌、局部最优 |
| rollout 长度 T | 24 | 太短：GAE 偏差大；太长：on-policy 数据陈旧 |
| 网络容量 | [128,64,32] | 过小欠拟合，过大浪费推理延迟 |

### Q37：如何在已有策略基础上做"带载重"的微调？要注意哪些坑？

**参考答案（区分度高，考察真实工程经验）：**

1. **明确目标工况**：如固定 +5kg 载荷（创建环境前设 `added_mass_range=[5,5]`）；
2. **加载权重 ≠ 恢复训练**：只加载 Actor/Critic 权重，**不恢复旧 Adam 状态**，固定一个较小 lr（adaptive 会立刻把"小学习率"改回去）；迭代计数清零，输出到独立日志目录，不覆盖源模型；
3. **深拷贝配置**：注册表返回的是共享配置对象，直接改会污染后续实验；
4. **公平评估**：三方对照——A 源策略零样本、B 同预算从零训练、C 微调；对齐交互预算、多种子、同一测试集；测新能力**和旧能力遗忘**（无载荷回归测试）；
5. **防遗忘**：新旧工况混合训练、降低 lr/epoch、周期性评估旧能力保留 Pareto 候选；注意 PPO 是 on-policy，**不能塞旧 rollout 防遗忘**，旧工况要用当前策略重采；
6. **Critic 适配**：新奖励分布下 value loss 可能暴涨，考虑重置/保留 Critic 的对照实验。

### Q38：什么是奖励黑客（reward hacking）？请结合本项目举例。

**参考答案：**

策略找到奖励函数的漏洞——拿高分但不完成真实任务。本项目中的典型例子：

- `feet_air_time` 鼓励腾空 → 策略可能学会原地小跳而不跟踪速度；
- 强 `torques` 惩罚 → 机器人长期压低重心、蠕动前进；
- 强 `lin_vel_z` 惩罚 + 弱跟踪奖励 → 站着不动反而划算（零速度误差比摔倒风险划算）；
- `action_rate` 惩罚过大 → 动作僵硬、响应迟钝。

防御：① 惩罚项加门控（如 `feet_air_time` 用非零指令门控）；② 不只看总 reward，**用独立原始指标验收**（速度 RMSE、摔倒率、滑移率）；③ 对奖励项做消融实验；④ `only_positive_rewards` 截断在一定程度上缓解"惩罚主导"。

### Q39：断点续训（resume）和加载预训练权重做微调，有什么区别？

**参考答案：**

- **Resume（`--resume --checkpoint N`）**：恢复 Actor/Critic 权重 **+ Adam 优化器状态 + 迭代计数**，目标是接着原训练继续跑，lr 等保持续训前状态（当前实现并非逐位精确恢复）；
- **微调初始化**：只把权重当初始点，优化器重置、lr 重新设定、迭代计数清零，目标分布/奖励通常有变化。

混淆两者的典型事故：想用小 lr 微调，结果 resume 把旧 Adam 的大 lr 惯性也带回来了。面试中这个区别几乎是必问。

---


## 十、编码与数学推导题

### Q40：手写 PPO 的核心损失计算（给定 new/old log_prob、advantages、values、returns）。

**参考答案：**

```python
import torch

def ppo_loss(new_log_prob, old_log_prob, advantages,
             values, old_values, returns, entropy,
             clip_param=0.2, value_loss_coef=1.0, entropy_coef=0.01):
    # 1) 概率比（log 相减再 exp，数值稳定）
    ratio = torch.exp(new_log_prob - old_log_prob)

    # 2) clipped surrogate（取 min = 取相反数的 max）
    surr1 = -advantages * ratio
    surr2 = -advantages * torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    surrogate_loss = torch.max(surr1, surr2).mean()

    # 3) clipped value loss
    v_clip = old_values + (values - old_values).clamp(-clip_param, clip_param)
    v_loss = torch.max((values - returns)**2, (v_clip - returns)**2).mean()

    # 4) 总损失（熵取负号 = 最大化熵）
    loss = surrogate_loss + value_loss_coef * v_loss - entropy_coef * entropy.mean()
    return loss
```

加分项：指出之后还要 `clip_grad_norm_(params, 1.0)`，且这套流程对同一批数据重复 `num_epochs × num_mini_batches` 次。

### Q41：手写 GAE 的反向递推（含 done 与 timeout 处理）。

**参考答案：**

```python
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """rewards/values/dones: [T, N, 1]，last_value: [N, 1]"""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):                       # 必须反向递推
        nextvalue = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t].float()              # 终止步不 bootstrap
        delta = rewards[t] + gamma * nextvalue * not_done - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * not_done * lastgaelam
    returns = advantages + values
    # 优势归一化（对整轮 rollout）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages
```

追问点：为什么 timeout 不能简单把 done 置 1？——见 Q24，超时是人为截断，真实状态仍有价值，应 bootstrap。

### Q42：推导对角高斯策略两个分布之间的 KL 散度（本项目 KL 自适应学习率用的公式）。

**参考答案：**

两个对角高斯 $\pi_{old}=\mathcal N(\mu_o, \sigma_o^2)$、$\pi_{new}=\mathcal N(\mu_n, \sigma_n^2)$，逐维：

$$D_{KL}(\pi_{old}\|\pi_{new}) = \sum_i \left[\log\frac{\sigma_n}{\sigma_o} + \frac{\sigma_o^2 + (\mu_o-\mu_n)^2}{2\sigma_n^2} - \frac{1}{2}\right]$$

代码（`ppo.py` 自适应分支）与此逐项对应，对 batch 取均值后与 `desired_kl=0.01` 比较调速。要点：① 方向是 $KL(old\|new)$；② 这是解析解，不需要采样估计；③ 各维求和，不是取平均。

### Q43：计算题——4096 环境、每环境 24 步、5 epoch、4 minibatch。一次迭代的样本数、每次梯度更新的 batch、总共多少次梯度更新？若每秒 20 万步采样，一次迭代采样耗时多少？

**参考答案：**

- 总样本 $B = 4096 \times 24 = 98304$；
- minibatch 大小 $= 98304 / 4 = 24576$；
- 梯度更新次数 $= 5 \times 4 = 20$ 次；
- 采样耗时 $= 98304 / 200000 \approx 0.49$ 秒。

追问：这些数据**不能跨迭代复用**（on-policy），每次迭代重新采集——再次强调采样吞吐的意义。

### Q44：手推策略梯度中"引入基线不改变期望"的证明。

**参考答案：**

$$\mathbb{E}_{a\sim\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s)\, b(s)\right] = b(s)\sum_a \pi_\theta(a|s)\nabla_\theta \log\pi_\theta(a|s) = b(s)\sum_a \nabla_\theta \pi_\theta(a|s) = b(s)\,\nabla_\theta \underbrace{\sum_a \pi_\theta(a|s)}_{=1} = 0$$

关键步骤用了 **log-derivative trick**：$\pi \nabla \log\pi = \nabla\pi$。所以加任何只依赖状态（不依赖动作）的基线都不引入偏差，但可降方差——这就是 Actor-Critic 用 $V(s)$ 当基线的理论依据。

### Q45：RolloutStorage 的张量为什么组织成 `[T, num_envs, dim]`（时间维在前）而不是 `[num_envs, T, dim]`？

**参考答案：**

1. **写入效率**：采样循环每一步拿到的是所有环境的一步数据 `[num_envs, dim]`，时间维在前时 `storage[t].copy_(transition)` 是连续内存写入；若环境维在前，每步要跨步长写入所有环境；
2. **GAE 反向递推**按时间维遍历，天然契合；
3. 更新时再统一展平 `[T×N, dim]` 打乱切 minibatch，两种需求都满足。

---

## 附录：高频追问速查

| 追问 | 一句话答案 |
|---|---|
| PPO 为什么比朴素 PG 稳？ | clip 限制单步更新幅度 + GAE 降方差 + 多 minibatch 复用 |
| 为什么不用 DQN？ | 动作是 12 维连续空间 |
| 为什么不用 SAC？ | on-policy PPO 与大规模并行采样契合，稳定可复现；SAC 对高维机器人状态的 replay buffer 工程更重 |
| 观测里为什么放上一帧动作？ | 提供动作时序信息，帮助平滑输出、感知延迟 |
| std 是网络输出的吗？ | 不是，是与状态无关的可学习参数（12 维） |
| 训练 300 次迭代总交互步数？ | $300 \times 98304 \approx 2.95\times10^7$ 步 |
| 熵系数为 0 会怎样？ | std 可能过早崩塌，策略陷入局部最优 |
| value loss 一直很大是发散吗？ | 不一定——回报目标随策略改进在移动，重点看是否 NaN/爆炸 |
| play.py 能当正式评估吗？ | 不能，它是定性回放（改了噪声/地形等配置），正式评估需固定工况、固定种子、原始指标 |
| 地形课程的升级依据？ | 机器人在该地形上走过的距离足够远 → 升级，过近 → 降级 |

---

> **使用建议**：先合上答案自测，能口述出要点 + 项目里的具体数值/文件位置，才算真正掌握。
> 追问时主动把话题引到代码细节（如"这个在 ppo.py 的 update() 里是这样实现的……"），是展示"真做过"的最强信号。

