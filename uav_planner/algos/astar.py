"""改进 A* 参考路径生成器 —— 论文 Sec. 3.2。

论文出处
--------
H. Wang et al., "AGP-MADDPG ...", Expert Systems With Applications 333 (2027).

Sec. 3.2 的改进 A* 在标准 A* 上加了两个多目标惩罚项：

    f_ours(n) = g(n) + h(n) + lambda_obs * c_obs(n) + lambda_sep * c_sep(n)   (2)
    c_obs(n)  = [max(0, 1 - d_obs(n)/d_safe)]^2                              (3)
    c_sep(n)  = sum_p exp(-||x(n) - p||^2 / (2 * sigma_sep^2))               (4)

消融变体作为独立算法注册
------------------------
论文 Sec. 4.5 用消融实验说明每个组件的贡献。为了让这些变体
也能在界面上直接选中、看到搜索过程，这里把 4 个变体都注册成
独立的算法：

    astar            完整改进 A*（硬约束 + c_obs + c_sep）
    astar-standard   标准 A*（三个组件全关）
    astar-hard       仅硬净空约束
    astar-hard-obs   硬约束 + c_obs

这样你在算法下拉框里切来切去，就能直观对比搜索过程。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..algo_registry import register_algo
from ..astar3d import AStar3D, CostWeights, PathResult
from ..environment import Cell, VoxelMap


def _run(world: VoxelMap, start: Cell, goal: Cell,
         weights: Optional[CostWeights],
         diagonal: bool, max_expansions: int, heuristic: str,
         safety_margin: Optional[float], other_paths: Sequence[np.ndarray],
         use_hard_constraint: bool,
         use_obstacle_cost: bool,
         use_separation_cost: bool) -> PathResult:
    """统一的执行入口，所有变体共用。"""
    planner = AStar3D(
        world,
        weights=weights or CostWeights(),
        diagonal=diagonal,
        max_expansions=max_expansions,
        heuristic=heuristic,
        safety_margin=safety_margin,
        use_hard_constraint=use_hard_constraint,
        use_obstacle_cost=use_obstacle_cost,
        use_separation_cost=use_separation_cost,
    )
    return planner.plan(start, goal, other_paths=list(other_paths))


# ============================================================ 完整方法
@register_algo("astar", "改进 A* · 完整（硬约束+c_obs+c_sep）")
def plan_astar(world: VoxelMap, start: Cell, goal: Cell,
               weights: Optional[CostWeights] = None,
               diagonal: bool = True,
               max_expansions: int = 500_000,
               heuristic: str = "euclidean",
               safety_margin: Optional[float] = None,
               other_paths: Sequence[np.ndarray] = (),
               **_ignored) -> PathResult:
    """论文 Sec. 3.2 的完整改进 A*（式 2/3/4 全启用）。"""
    return _run(world, start, goal, weights, diagonal, max_expansions,
                heuristic, safety_margin, other_paths,
                use_hard_constraint=True,
                use_obstacle_cost=True,
                use_separation_cost=True)


# ============================================================ 消融变体
@register_algo("astar-standard", "消融 · 标准 A*（无任何改进）")
def plan_astar_standard(world: VoxelMap, start: Cell, goal: Cell,
                        weights: Optional[CostWeights] = None,
                        diagonal: bool = True,
                        max_expansions: int = 500_000,
                        heuristic: str = "euclidean",
                        safety_margin: Optional[float] = None,
                        other_paths: Sequence[np.ndarray] = (),
                        **_ignored) -> PathResult:
    """消融：去掉硬约束、c_obs、c_sep，退化为标准 A*。

    注意：没有硬约束时，路径可能贴着甚至穿过障碍，
    这正是消融实验要展示的效果。
    """
    return _run(world, start, goal, weights, diagonal, max_expansions,
                heuristic, safety_margin, other_paths,
                use_hard_constraint=False,
                use_obstacle_cost=False,
                use_separation_cost=False)


@register_algo("astar-hard", "消融 · +硬净空约束")
def plan_astar_hard(world: VoxelMap, start: Cell, goal: Cell,
                    weights: Optional[CostWeights] = None,
                    diagonal: bool = True,
                    max_expansions: int = 500_000,
                    heuristic: str = "euclidean",
                    safety_margin: Optional[float] = None,
                    other_paths: Sequence[np.ndarray] = (),
                    **_ignored) -> PathResult:
    """消融：只加硬净空约束，不加 c_obs / c_sep。"""
    return _run(world, start, goal, weights, diagonal, max_expansions,
                heuristic, safety_margin, other_paths,
                use_hard_constraint=True,
                use_obstacle_cost=False,
                use_separation_cost=False)


@register_algo("astar-hard-obs", "消融 · +硬约束+障碍代价")
def plan_astar_hard_obs(world: VoxelMap, start: Cell, goal: Cell,
                        weights: Optional[CostWeights] = None,
                        diagonal: bool = True,
                        max_expansions: int = 500_000,
                        heuristic: str = "euclidean",
                        safety_margin: Optional[float] = None,
                        other_paths: Sequence[np.ndarray] = (),
                        **_ignored) -> PathResult:
    """消融：硬约束 + 式(3) 障碍代价，不加 c_sep。"""
    return _run(world, start, goal, weights, diagonal, max_expansions,
                heuristic, safety_margin, other_paths,
                use_hard_constraint=True,
                use_obstacle_cost=True,
                use_separation_cost=False)
