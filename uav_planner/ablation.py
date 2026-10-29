"""消融实验（Ablation study）—— 对应论文 Sec. 4.5。

论文的消融设计
--------------
论文在 Sec. 4.5 比较了 4 个变体：

    MADDPG                    完整基线（无引导、无注意力）
    AGP-MADDPG w/o Attention  去掉注意力 critic
    AGP-MADDPG w/o Guidance   去掉 A* 引导点奖励
    AGP-MADDPG                完整方法

本项目前期只复现**规划层**（论文 Sec. 3.2 的改进 A*），
所以消融针对 A* 的三个组件：

    1. 硬净空约束（hard constraint）  —— 论文用软惩罚，这里额外保留硬下限
    2. 障碍代价 c_obs                 —— 式 (3)
    3. 机间分离代价 c_sep             —— 式 (4)

论文报告的指标（Table 6/7）
--------------------------
    Average decision steps   平均决策步数（越小越好）
    Mean trajectory length   平均轨迹长度（越小越好）
    Mission success rate     任务成功率（越大越好）

本项目对应的规划层指标
--------------------
    path_length      路径长度（对应 Mean trajectory length）
    waypoints        航点数（近似 decision steps）
    min_clearance    最小净空
    clearance_violation 是否违反硬净空
    expanded         扩展节点数
    success_rate     规划成功率
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import PlannerConfig, ScenarioConfig
from .fleet import FleetPlanner
from .metrics import evaluate_plan


@dataclass
class AblationVariant:
    """一个消融变体。"""

    name: str
    description: str
    use_hard_constraint: bool = True
    use_obstacle_cost: bool = True
    use_separation_cost: bool = True


# 论文 Sec. 4.5 的对照思路映射到规划层
DEFAULT_VARIANTS: List[AblationVariant] = [
    AblationVariant(
        name="standard-astar",
        description="标准 A*（无硬约束、无障碍代价、无分离代价）",
        use_hard_constraint=False,
        use_obstacle_cost=False,
        use_separation_cost=False,
    ),
    AblationVariant(
        name="+hard",
        description="加硬净空约束",
        use_hard_constraint=True,
        use_obstacle_cost=False,
        use_separation_cost=False,
    ),
    AblationVariant(
        name="+hard+obs",
        description="加硬约束 + 障碍代价 c_obs（式 3）",
        use_hard_constraint=True,
        use_obstacle_cost=True,
        use_separation_cost=False,
    ),
    AblationVariant(
        name="full",
        description="完整改进 A*（硬约束 + c_obs + c_sep）",
        use_hard_constraint=True,
        use_obstacle_cost=True,
        use_separation_cost=True,
    ),
]


def _run_variant(scenario, variant: AblationVariant, planner_cfg: PlannerConfig) -> Dict:
    """跑一个变体，返回指标。"""
    from .astar3d import AStar3D

    world = scenario.world
    weights = planner_cfg.to_weights()
    starts = [world.world_to_grid(p) for p in scenario.starts]
    goals = [world.world_to_grid(p) for p in scenario.goals]

    planner = AStar3D(
        world,
        weights=weights,
        diagonal=planner_cfg.diagonal_cost,
        max_expansions=planner_cfg.max_expansions,
        heuristic=planner_cfg.heuristic,
        safety_margin=planner_cfg.safety_margin,
        use_hard_constraint=variant.use_hard_constraint,
        use_obstacle_cost=variant.use_obstacle_cost,
        use_separation_cost=variant.use_separation_cost,
    )

    results = []
    for i, (s, g) in enumerate(zip(starts, goals)):
        previous = [r.points for r in results]
        try:
            results.append(planner.plan(s, g, previous))
        except Exception as exc:
            return {
                "variant": variant.name,
                "description": variant.description,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    raw_paths = [np.asarray(r.points, dtype=float) for r in results]

    # 平滑必须和主流程（FleetPlanner._post_process）完全一致：
    # 平滑后如果净空低于硬约束，就回退到原始路径。
    # 少了这个校验，平滑会把"硬约束带来的安全余量"抹平，
    # 于是被约束的变体看起来反而不如标准 A*。
    hard = weights.hard_clearance
    from .smoothing import smooth_collision_aware

    def post(points):
        if len(points) < 3:
            return points.copy()
        candidate = smooth_collision_aware(points, world, safety_radius=hard)
        if world.path_clearance(candidate) >= hard - 1e-9:
            return candidate
        return points.copy()

    paths = [post(p) for p in raw_paths]

    def length_of(p):
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0

    raw_lengths = [length_of(p) for p in raw_paths]
    raw_clearances = [world.path_clearance(p) for p in raw_paths]
    lengths = [length_of(p) for p in paths]
    clearances = [world.path_clearance(p) for p in paths]

    return {
        "variant": variant.name,
        "description": variant.description,
        "success": True,
        # 平滑后的终态（和右侧面板同一口径）
        "path_length": float(np.mean(lengths)),
        "waypoints": float(np.mean([len(p) for p in paths])),
        "min_clearance": float(np.min(clearances)),
        "clearance_violation": bool(np.min(clearances) < hard - 1e-9),
        # 规划器原始输出（不受平滑影响，最能反映各变体的本质差异）
        "raw_path_length": float(np.mean(raw_lengths)),
        "raw_min_clearance": float(np.min(raw_clearances)),
        "raw_clearance_violation": bool(np.min(raw_clearances) < hard - 1e-9),
        "expanded": int(sum(r.expanded for r in results)),
        "success_rate": 1.0,
    }


def run_ablation(config: Optional[ScenarioConfig] = None,
                 variants: Optional[List[AblationVariant]] = None,
                 map_name: str = "walls",
                 seeds: Optional[List[int]] = None) -> Dict:
    """跑完整消融实验。

    Parameters
    ----------
    config   : 场景配置（seed 会被 seeds 覆盖）
    variants : 消融变体列表，默认用 DEFAULT_VARIANTS
    map_name : 地图名
    seeds    : 多个随机种子，结果取均值（论文用 N_test = 5）

    Returns
    -------
    dict，包含每个变体在每个 seed 上的指标，以及汇总均值。
    """
    from . import registry

    config = config or ScenarioConfig()
    variants = variants or DEFAULT_VARIANTS
    seeds = seeds or [config.seed]

    rows: List[Dict] = []
    for variant in variants:
        per_seed = []
        for seed in seeds:
            cfg = ScenarioConfig(
                seed=seed,
                n_uavs=config.n_uavs,
                n_obstacles=config.n_obstacles,
                planner=config.planner,
            )
            scenario = registry.build(map_name, cfg)
            per_seed.append(_run_variant(scenario, variant, cfg.planner))

        ok = [r for r in per_seed if r.get("success")]
        if not ok:
            rows.append({
                "variant": variant.name,
                "description": variant.description,
                "success": False,
                "error": per_seed[0].get("error", "unknown"),
            })
            continue

        rows.append({
            "variant": variant.name,
            "description": variant.description,
            "success": True,
            # 平滑后（与右侧面板同口径）
            "path_length": float(np.mean([r["path_length"] for r in ok])),
            "path_length_std": float(np.std([r["path_length"] for r in ok])),
            "waypoints": float(np.mean([r["waypoints"] for r in ok])),
            "min_clearance": float(np.min([r["min_clearance"] for r in ok])),
            "clearance_violation": any(r["clearance_violation"] for r in ok),
            # 规划器原始输出（不受平滑影响）
            "raw_path_length": float(np.mean([r["raw_path_length"] for r in ok])),
            "raw_min_clearance": float(np.min([r["raw_min_clearance"] for r in ok])),
            "raw_clearance_violation": any(r["raw_clearance_violation"] for r in ok),
            "expanded": float(np.mean([r["expanded"] for r in ok])),
            "success_rate": float(np.mean([1.0 if r.get("success") else 0.0 for r in per_seed])),
        })

    return {
        "map": map_name,
        "seeds": list(seeds),
        "n_uavs": config.n_uavs,
        "rows": rows,
    }


def format_ablation(report: Dict) -> str:
    """把消融结果格式化成表格（对齐论文 Table 6/7 的样式）。"""
    lines = [
        f"消融实验 · map={report['map']} · uavs={report['n_uavs']} · "
        f"seeds={report['seeds']}",
        "",
        f"{'变体':<18} {'路径长度':>10} {'航点':>8} {'最小净空':>10} "
        f"{'扩展数':>10} {'成功率':>8} {'违规':>6}",
        "-" * 78,
    ]
    for row in report["rows"]:
        if not row.get("success"):
            lines.append(f"{row['variant']:<18} 失败: {row.get('error', '')}")
            continue
        lines.append(
            f"{row['variant']:<18} "
            f"{row['path_length']:>10.2f} "
            f"{row['waypoints']:>8.1f} "
            f"{row['min_clearance']:>10.2f} "
            f"{row['expanded']:>10.0f} "
            f"{row['success_rate']:>7.0%} "
            f"{'是' if row['clearance_violation'] else '否':>6}"
        )
    return "\n".join(lines)
