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
# 【模块说明】配置基类文件。定义 BaseConfig，其核心作用是：把配置类中
# 以"嵌套 class"形式声明的成员（如 class env、class terrain）递归地实例化，
# 使得外部代码可以用 cfg.env.num_envs 这种属性链方式直接读取配置值。
# 项目中所有配置类（如 LeggedRobotCfg、LeggedRobotCfgPPO）都继承自它。
# =====================================================================

import inspect

# 【类说明】BaseConfig：所有配置类的基类。
# 设计技巧：配置项用"类中嵌套类"的方式声明（类即命名空间），
# 本类在 __init__ 时递归地把这些嵌套类替换成实例，从而支持属性式访问。
class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        # 【中文】构造时调用递归初始化，把所有嵌套的成员类实例化
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # 【中文】遍历对象的所有属性名（包括数据属性和嵌套类）
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            # 【中文】跳过内置属性 __class__，避免把类型本身实例化造成死循环
            if key=="__class__":
                continue
            # 【中文】取出该属性对应的对象（可能是普通值，也可能是一个类）
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # 【中文】判断该属性是否是一个"类"（嵌套配置类即为此情况）
            # check if it the attribute is a class
            if inspect.isclass(var):
                # 【中文】调用无参构造函数，把嵌套类实例化
                # instantate the class
                i_var = var()
                # 【中文】关键一步：用实例替换原来的类对象，
                # 此后 obj.key 访问到的是实例属性而不是类属性
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # 【中文】递归处理该实例内部的嵌套类，实现任意深度的配置树
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)