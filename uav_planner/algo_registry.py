
"""算法注册表 —— 插件式规划器系统。

和 registry.py 同一套设计。你以后加算法，只需要：

    1. 在 uav_planner/algos/ 下新建一个 .py 文件
    2. 写一个 `register_algo("名字") 装饰的函数，返回 PathResult
    3. 什么都不用改

规划器契约（必须遵守）
----------------------
签名：`plan(world, start, goal, weights, **kwargs) -> PathResult`

    world   : VoxelMap          体素地图
    start   : (i,j,k) int 元组  起点格子索引
    goal    : (i,j,k) int 元组  终点格子索引
    weights : CostWeights       代价权重（含 hard_clearance 硬约束）
    kwargs  : 可选，会收到 diagonal / max_expansions / heuristic /
              safety_margin / other_paths / seed

返回：PathResult(points=(N,3) 世界坐标, cost, expanded, planner, metadata)

参考实现见 algos/astar.py。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .astar3d import CostWeights, PathResult
from .environment import Cell, VoxelMap

# 规划器签名
AlgoFn = Callable[..., PathResult]

_ALGOS: Dict[str, AlgoFn] = {}
_LABELS: Dict[str, str] = {}


def register_algo(name: str, label: Optional[str] = None):
    """装饰器：把一个规划函数注册到全局表。

    用法::

        `register_algo("astar", "A* 安全多目标")
        def plan_astar(world, start, goal, weights, **kw):
            ...

    重复注册同名会抛 ValueError。
    """
    def decorator(fn: AlgoFn) -> AlgoFn:
        if name in _ALGOS:
            raise ValueError(f"algorithm '{name}' is already registered")
        _ALGOS[name] = fn
        _LABELS[name] = label or name
        return fn
    return decorator


def get_algo(name: str) -> AlgoFn:
    """按名字取规划函数，找不到抛 KeyError（附可用列表）。"""
    try:
        return _ALGOS[name]
    except KeyError:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(_ALGOS)}") from None


def has_algo(name: str) -> bool:
    return name in _ALGOS


def list_algos() -> tuple:
    return tuple(_ALGOS)


def algo_labels() -> Dict[str, str]:
    return dict(_LABELS)


def plan(name: str, world: VoxelMap, start: Cell, goal: Cell,
         weights: Optional[CostWeights] = None, **kwargs) -> PathResult:
    """按名字调用规划器。"""
    return get_algo(name)(world, start, goal, weights or CostWeights(), **kwargs)


def plan_fleet(name: str, world: VoxelMap, starts, goals,
               weights: Optional[CostWeights] = None, **kwargs):
    """多机优先级规划：每架避开前面已经规划好的路径。

    这是通用的多机策略，所有算法都能用 —— 它只依赖 PathResult，
    不关心内部是 A* 还是 RRT*。
    """
    weights = weights or CostWeights()
    fn = get_algo(name)
    results = []
    for start, goal in zip(starts, goals):
        previous = [r.points for r in results]
        results.append(fn(world, start, goal, weights,
                          other_paths=previous, **kwargs))
    return results


def load_plugins() -> None:
    """自动导入 algos/ 目录下所有模块，触发装饰器注册。"""
    import importlib
    import pkgutil

    package = importlib.import_module(__package__ + ".algos")
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{package.__name__}.{info.name}")


__all__ = [
    "AlgoFn", "register_algo", "get_algo", "has_algo", "list_algos",
    "algo_labels", "plan", "plan_fleet", "load_plugins",
    "PathResult", "CostWeights", "VoxelMap", "Cell",
]
