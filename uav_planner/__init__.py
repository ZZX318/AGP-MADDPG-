
"""UAV 3D Planner 模板工程 —— 可扩展的三维多无人机路径规划。

目录结构
--------
uav_planner/
    registry.py        地图注册表（加地图看这里）
    algo_registry.py   算法注册表（加算法看这里）
    scenario.py        场景公共零件（端点采样、连通性验证）
    environment.py     体素地图、净空栅格、passable 掩码
    geometry.py        解析几何（球/盒/柱 + 连续碰撞检查）
    config.py          全部参数（唯一真源）
    astar3d.py         改进 A* 实现（论文 Sec. 3.2）
    smoothing.py       碰撞感知平滑
    guidance.py        动态引导点
    dynamics.py        限速限加速模型
    fleet.py           组合层：规划 -> 平滑 -> rollout
    metrics.py         规划层/执行层双份指标
    cli.py             命令行入口
    maps/              地图插件（加地图在这里新建文件）
    algos/             算法插件（加算法在这里新建文件）

加地图
------
在 uav_planner/maps/ 下新建 .py，写：

    from ..registry import register_map

    @register_map("mymap", "我的地图")
    def make_mymap(config=None):
        ...

加算法
------
在 uav_planner/algos/ 下新建 .py，写：

    from ..algo_registry import register_algo

    @register_algo("myalgo", "我的算法")
    def plan_myalgo(world, start, goal, weights, **kwargs):
        ...

两个注册表都是自动发现，写完不用改任何其它文件。
"""

from .config import CostWeights, DynamicsConfig, PlannerConfig, RewardConfig, ScenarioConfig
from .environment import VoxelMap
from .astar3d import AStar3D, PathResult
from .fleet import FleetPlan, FleetPlanner
from .guidance import GuidanceTracker
from .dynamics import KinematicModel, UAVState
from .metrics import evaluate, evaluate_plan, format_metrics, format_report
from .scenario import Scenario

# 注册表
from . import registry, algo_registry

# 自动加载插件（maps/ 和 algos/ 下所有模块）
registry.load_plugins()
algo_registry.load_plugins()


def maps():
    """所有可用地图名。"""
    return registry.list_maps()


def algos():
    """所有可用算法名。"""
    return algo_registry.list_algos()


def make_scenario(name, config=None):
    """按名字构造场景。"""
    return registry.build(name, config)


def plan_fleet(algorithm, world, starts, goals, weights=None, **kwargs):
    """按名字做多机规划。"""
    return algo_registry.plan_fleet(algorithm, world, starts, goals, weights, **kwargs)


__all__ = [
    "AStar3D", "CostWeights", "DynamicsConfig", "FleetPlan", "FleetPlanner",
    "GuidanceTracker", "KinematicModel", "PathResult", "PlannerConfig",
    "RewardConfig", "Scenario", "ScenarioConfig", "UAVState", "VoxelMap",
    "evaluate", "evaluate_plan", "format_metrics", "format_report",
    "registry", "algo_registry", "maps", "algos", "make_scenario", "plan_fleet",
]
