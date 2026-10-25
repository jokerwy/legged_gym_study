# =====================================================================
# 【模块说明】本文件是 legged_gym 项目的安装脚本（setup.py）。
# 通过 setuptools 将 legged_gym 打包为可安装的 Python 包，
# 执行 `pip install -e .` 后即可在任意位置 import legged_gym。
# =====================================================================
from setuptools import find_packages
from distutils.core import setup

# 调用 setup() 定义包的元数据与依赖关系
# find_packages() 会自动查找当前目录下所有含 __init__.py 的子包并一并安装
setup(
    # 包名：安装后通过 `import legged_gym` 使用
    name='legged_gym',
    # 版本号
    version='1.0.0',
    # 作者
    author='Nikita Rudin',
    # 开源协议：BSD 3 条款
    license="BSD-3-Clause",
    # 自动发现并包含所有子包（legged_gym.envs、legged_gym.utils 等）
    packages=find_packages(),
    # 作者邮箱
    author_email='rudinn@ethz.ch',
    # 包描述：基于 Isaac Gym 的腿式机器人仿真环境
    description='Isaac Gym environments for Legged Robots',
    # 安装依赖：Isaac Gym 仿真器、rsl_rl 强化学习库、matplotlib 绘图库
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib']
)