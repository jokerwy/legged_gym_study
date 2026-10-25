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
# 【模块说明】本文件实现地形生成器 Terrain，负责为腿式机器人仿真创建复杂地形。
# 核心思路：把整个世界划分成 num_rows x num_cols 个“子地形”网格，
# 每个子地形是一个独立的小关卡（斜坡/台阶/离散障碍/踏脚石/沟壑/深坑等），
# 全部写入一张大的高度场（heightfield）数组；最后可按需转换成三角网格
# （trimesh）交给 Isaac Gym 渲染与碰撞。
# 配合课程学习（curriculum）：行号 i 决定难度（越往下越难），列号 j 决定地形类型。
# 文件末尾还定义了两个自定义地形函数：gap_terrain（沟壑）与 pit_terrain（深坑）。
# =====================================================================
import numpy as np
from numpy.random import choice
from scipy import interpolate

# Isaac Gym 官方地形工具库：提供 SubTerrain 类与各种地形生成函数
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

# Terrain 类：按配置生成整块训练地形。
# 生成结果包括：
#   - height_field_raw：整块世界的高度场（int16，单位 = vertical_scale 米）；
#   - env_origins：每个子地形的“出生点”世界坐标（机器人初始放置位置）；
#   - vertices / triangles：当 mesh_type 为 "trimesh" 时的三角网格顶点与面片。
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        # 中文：构造函数。
        # 参数：cfg - 地形配置（LeggedRobotCfg.terrain，含网格类型、尺寸、
        # 子地形行/列数、各类地形比例等）；num_robots - 机器人数量。

        self.cfg = cfg
        self.num_robots = num_robots
        # 中文：地形网格类型："none"/"plane" 不生成起伏地形；
        # "heightfield" 用高度场；"trimesh" 转换为三角网格（最常用）
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            # 中文：平地场景无需任何地形生成，直接返回
            return
        # 中文：单个子地形的物理尺寸（米）
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        # 中文：把各类地形的比例列表转成“累积比例”列表（前缀和），
        # 之后用 [0,1) 均匀随机数落在哪个累积区间来决定生成哪种地形
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # 中文：子地形总数 = 行数 x 列数（num_rows 个难度等级 x num_cols 种地形类型）
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # 中文：每个子地形的出生点（世界坐标 x,y,z），形状 (num_rows, num_cols, 3)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # 中文：单个子地形在高度场中的像素尺寸 = 物理尺寸 / 水平分辨率（米/像素）
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # 中文：四周留白的边界宽度（像素），防止机器人生成在世界边缘
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        # 中文：整块高度场的总像素尺寸 = 所有子地形 + 两侧边界
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 中文：初始化整块高度场为 0（int16 类型，实际高度 = 像素值 x vertical_scale 米）
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        # 中文：三种地形排布模式，按配置三选一：
        if cfg.curriculum:
            # 中文：课程学习模式——行方向难度递增，列方向固定地形类型，地形难度系统化递进
            self.curiculum()
        elif cfg.selected:
            # 中文：指定地形模式——所有子地形都生成同一种由 terrain_kwargs 指定的地形
            self.selected_terrain()
        else:    
            # 中文：完全随机模式——每个子地形的类型与难度都随机抽取
            self.randomized_terrain()   
        
        # 中文：heightsamples 供高度扫描等传感器采样使用（此处直接引用整块高度场）
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            # 中文：把高度场转换成三角网格（顶点 + 三角形面片索引），供 Isaac Gym
            # 作为静态网格资产加入仿真。horizontal/vertical_scale 为水平/垂直分辨率，
            # slope_treshold 为坡度阈值（超过该坡度的面片会被标记/特殊处理）
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        # 中文：完全随机模式——遍历所有子地形，每个子地形的“类型”和“难度”都随机。
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            # 中文：把一维序号 k 换算成二维网格坐标 (i 行, j 列)
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # 中文：choice 为 [0,1) 均匀随机数，决定地形类型（配合累积比例 proportions）
            choice = np.random.uniform(0, 1)
            # 中文：difficulty 从三档难度中随机抽取（0.5 / 0.75 / 0.9，越大越难）
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            # 中文：生成该子地形并拼入整块高度场
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        # 中文：课程学习模式——子地形按网格系统化排布：
        #   行 i（0..num_rows-1）控制难度：difficulty = i / num_rows，越往下越难；
        #   列 j（0..num_cols-1）控制类型：choice = j / num_cols，各列对应不同地形类型。
        # 机器人训练时可根据表现从简单行“升级”到困难行，实现循序渐进的课程。
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # 中文：难度随行号线性增加（0 ~ 接近 1）
                difficulty = i / self.cfg.num_rows
                # 中文：类型随列号变化，+0.001 避免恰好取到 0 导致边界歧义
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        # 中文：指定地形模式——用 cfg.terrain_kwargs 中指定的同一种地形函数
        # 填充所有子地形（常用于评估/调试特定地形）。
        # 中文：弹出 'type' 键得到地形函数名（字符串），其余键作为该函数的参数
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            # 中文：一维序号换算成二维网格坐标
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # 中文：创建一块空白子地形画布（SubTerrain 对象）
            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            # 中文：用 eval 把字符串形式的地形函数名转成函数并调用，传入剩余参数
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        # 中文：根据类型选择参数 choice（落在哪个累积比例区间）和难度 difficulty
        # 生成一块具体的子地形。difficulty 越大，坡越陡、台阶越高、坑越深。
        # 参数：choice - [0,1) 区间内的类型选择值；difficulty - [0,1] 难度系数。
        # 返回：生成好的 terrain_utils.SubTerrain 对象。
        # 中文：先创建一块空白子地形画布，尺寸为单个子地形的像素宽/长，
        # 并携带垂直/水平分辨率信息
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        # 中文：各地形参数均随难度缩放——
        # 斜坡坡度（最大 0.4）；台阶高度（0.05 ~ 0.23 米）；
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        # 离散障碍高度（0.05 ~ 0.25 米）；踏脚石尺寸（难度越大石头越小）；
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        # 踏脚石间距；沟壑宽度（0 ~ 1 米）；深坑深度（0 ~ 1 米）
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        # 中文：按累积比例区间选择地形类型（choice 落在哪个区间就生成哪种）：
        if choice < self.proportions[0]:
            # 中文：类型1——金字塔斜坡（中间高四周低的锥形坡，中央留平台）。
            # 前一半概率取负坡度（下坡/凹坑），后一半为正坡度（上坡）
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            # 中文：类型2——斜坡 + 均匀随机粗糙地面（在斜坡上叠加小幅随机起伏，
            # 模拟野外粗糙地表）
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            # 中文：类型3——金字塔楼梯（向中心逐级升高的台阶）。
            # 落在 proportions[2] 之前的取负台阶高度（下行楼梯）
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            # 中文：类型4——离散障碍物：随机散布 20 个 1~2 米大小的矩形矮台，
            # 高度随难度增加，训练机器人跨越/绕开障碍
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            # 中文：类型5——踏脚石：散布一组高出地面的小平台，石头之间是空隙，
            # 训练机器人精确的落足点控制
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            # 中文：类型6——沟壑：环形深沟围住中央平台（本文件末尾自定义函数）
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            # 中文：类型7——深坑：中央一个大坑，四周是平台（本文件末尾自定义函数）
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        # 中文：把生成好的单个子地形写入整块高度场的对应网格位置，
        # 并计算该子地形的出生点（env_origin）存入 env_origins。
        # 参数：terrain - SubTerrain 对象；row/col - 子地形所在的行/列。
        i = row
        j = col
        # map coordinate system
        # 中文：计算该子地形在整块高度场中的像素范围（跳过边界 border，按行列偏移）
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        # 中文：把子地形的高度数据拷贝进总高度场对应区域
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # 中文：出生点的 x/y 取子地形中心的世界坐标
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # 中文：取子地形中心 2m x 2m 区域的像素索引范围
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        # 中文：出生点高度 z 取中心区域的最大高度（像素值 x 垂直分辨率），
        # 保证机器人出生时站在中央平台顶面上而不是陷进地形里
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    # 中文：生成“沟壑”地形——中央是平台，平台外环绕一条深沟（环形），
    # 再往外恢复为平地。机器人必须越过深沟才能离开中央平台。
    # 参数：terrain - SubTerrain 画布；gap_size - 沟宽（米）；
    # platform_size - 中央平台边长（米）。
    # 中文：把物理尺寸换算成像素数（除以水平分辨率）
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    # 中文：子地形中心像素坐标
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    # 中文：x1/y1 = 平台半边长（像素）；x2/y2 = 平台半边长 + 沟宽（像素），
    # 即沟壑的外沿位置
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    # 中文：先把外沿范围内整片设为极深（-1000 像素 x 垂直分辨率，近似“深渊”）
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    # 中文：再把中央平台区域填回 0，最终形成“平台 + 环形深沟”的结构
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    # 中文：生成“深坑”地形——中央一个指定深度的方形坑，四周是平台。
    # 参数：terrain - SubTerrain 画布；depth - 坑深（米）；
    # platform_size - 坑的边长（米，注意下面除以 2 得到半边长）。
    # 中文：深度换算成高度场像素值（除以垂直分辨率）
    depth = int(depth / terrain.vertical_scale)
    # 中文：坑的半边长换算成像素（除以水平分辨率再除以 2）
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    # 中文：坑区域在 x/y 方向的像素范围（以子地形中心为中心）
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    # 中文：把坑区域高度设为 -depth，形成方坑
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
