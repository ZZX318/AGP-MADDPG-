
"""场景基础设施 —— 端点采样、连通性验证、修复。

这个文件只保留"造场景的公共零件"，具体地图见 maps/ 目录。
所有地图插件都应该复用这里的 _sample_endpoints 和 _finish。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import ScenarioConfig
from .environment import VoxelMap


@dataclass
class Scenario:
    """一张完整的实验场景。"""
    world: VoxelMap
    starts: list
    goals: list
    info: dict = field(default_factory=dict)


def _sample_endpoints(world: VoxelMap, rng: np.random.Generator, n_uavs: int,
                      clearance: float, margin: float):
    """在整个三维体积里采起终点对。

    设计要点（来自原项目，很重要）：
      * 端点在 x/y/z 三个方向都随机，所以路径不是被困在一个 z 层里
      * 起点在一侧、终点在另一侧，中间必然要穿越障碍
      * 每个端点都吸附到最近的可通行格，保证不落在障碍里
    """
    nx, ny, nz = world.shape
    flip = bool(rng.integers(2))                      # 种子决定方向
    x_start = int(rng.integers(1, 3)) if not flip else int(rng.integers(nx - 3, nx - 1))
    x_goal = int(rng.integers(nx - 3, nx - 1)) if not flip else int(rng.integers(1, 3))

    # 把 y 轴切成 n_uavs 条车道，让机队分散开
    lane = (ny - 4) / max(n_uavs, 1)
    ys_start = [int(np.clip(2 + lane * (i + 0.5) + rng.uniform(-0.35, 0.35) * lane,
                            1, ny - 2)) for i in range(n_uavs)]
    ys_goal = [int(np.clip(2 + lane * (i + 0.5) + rng.uniform(-0.35, 0.35) * lane,
                           1, ny - 2)) for i in range(n_uavs)]
    zs_start = [int(rng.integers(1, nz - 1)) for _ in range(n_uavs)]
    zs_goal = [int(rng.integers(1, nz - 1)) for _ in range(n_uavs)]

    starts, goals = [], []
    for i in range(n_uavs):
        s = world.nearest_passable(world.grid_to_world((x_start, ys_start[i], zs_start[i])),
                                   clearance, margin)
        g = world.nearest_passable(world.grid_to_world((x_goal, ys_goal[i], zs_goal[i])),
                                   clearance, margin)
        starts.append(world.grid_to_world(s))
        goals.append(world.grid_to_world(g))
    return starts, goals


def _finish(world: VoxelMap, starts, goals, config: ScenarioConfig, layout: str,
            repairs: int) -> Scenario:
    """验证所有起终点对连通，不通就删障碍重试。

    这是"生成 → 验证 → 修复"的最后一步，永远不要跳过它，
    否则规划阶段会抛 RuntimeError: no feasible 3D path found。
    """
    clearance = float(config.planner.hard_clearance)
    margin = (0.5 * np.sqrt(3.0) * config.planner.resolution
              if config.planner.safety_margin is None else float(config.planner.safety_margin))

    while repairs <= config.max_repairs and world.obstacles:
        if all(world.reachable(s, g, clearance, margin) for s, g in zip(starts, goals)):
            break
        world.remove_obstacle()
        repairs += 1

    if world.obstacles and not all(world.reachable(s, g, clearance, margin)
                                   for s, g in zip(starts, goals)):
        world.clear_obstacles()      # 最后手段：清空障碍，永不 raise

    info = {
        "seed": config.seed,
        "obstacles": len(world.obstacles),
        "requested_obstacles": config.n_obstacles,
        "repairs": repairs,
        "occupancy": float(world.occupied.mean()),
        "clearance": clearance,
        "margin": margin,
        "layout": layout,
    }
    return Scenario(world, starts, goals, info)


__all__ = ["Scenario", "_sample_endpoints", "_finish"]
