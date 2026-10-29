
"""命令行入口。

所有地图和算法都从注册表动态读取，加插件不用改这个文件。
"""

from __future__ import annotations

import argparse
import io
import json
import time

import numpy as np

from . import algo_registry, registry
from .config import PlannerConfig, ScenarioConfig
from .dynamics import KinematicModel
from .fleet import FleetPlan, FleetPlanner
from .metrics import evaluate_plan, format_report


# --------------------------------------------------------------------- 参数
def build_planner_config(args) -> PlannerConfig:
    """命令行参数 -> PlannerConfig。"""
    return PlannerConfig(
        resolution=args.resolution,
        shape=tuple(args.shape),
        max_expansions=args.max_expansions,
        diagonal_cost=not args.no_diagonal,
        heuristic=args.heuristic,
        safety_margin=args.margin,
        hard_clearance=args.hard,
        clearance=args.clearance,
        minimum_separation=args.min_separation,
    )


def build_scenario_config(args) -> ScenarioConfig:
    return ScenarioConfig(
        seed=args.seed,
        n_uavs=args.uavs,
        n_obstacles=args.obstacles,
        planner=build_planner_config(args),
    )


# --------------------------------------------------------------------- 规划
def run_plan(scenario, algorithm: str, args, planner_cfg: PlannerConfig):
    """跑一次完整流程：规划 -> 平滑 -> rollout。返回 (FleetPlan, timings)。"""
    weights = planner_cfg.to_weights()
    starts = [scenario.world.world_to_grid(p) for p in scenario.starts]
    goals = [scenario.world.world_to_grid(p) for p in scenario.goals]

    t0 = time.perf_counter()
    results = algo_registry.plan_fleet(
        algorithm, scenario.world, starts, goals, weights,
        diagonal=not args.no_diagonal,
        max_expansions=args.max_expansions,
        heuristic=args.heuristic,
        safety_margin=args.margin,
        seed=args.seed,
    )
    planning_time = time.perf_counter() - t0

    # 平滑（碰撞感知，失败则回退原路径）
    fleet = FleetPlanner(scenario.world, weights)
    paths = [fleet._post_process(r.points, smooth=not args.no_smooth) for r in results]

    plan = FleetPlan(paths=paths, results=results)

    t0 = time.perf_counter()
    fleet.rollout(plan, scenario.goals, KinematicModel(),
                  hard_clearance=weights.hard_clearance)
    rollout_time = time.perf_counter() - t0

    return plan, {"planning": planning_time, "rollout": rollout_time}


# --------------------------------------------------------------------- 输出
def _json_safe(value):
    """JSON 没有 Infinity，把 inf 换成 null。"""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def describe(scenario, plan, report, timings, args) -> str:
    info = scenario.info
    lines = [
        f"map={args.map}, algorithm={args.algorithm}, uavs={args.uavs}, "
        f"obstacles={info.get('obstacles')}/{info.get('requested_obstacles')} "
        f"(repairs={info.get('repairs')}), seed={args.seed}, "
        f"occupancy={info.get('occupancy', 0.0):.1%}",
        f"plan={timings.get('planning', 0.0):.2f}s rollout={timings.get('rollout', 0.0):.2f}s",
        "",
        format_report(report),
        "",
    ]
    for i, result in enumerate(plan.results):
        meta = result.metadata
        lines.append(
            f"UAV {i}: {result.planner}, waypoints={len(plan.paths[i])}, "
            f"cost={result.cost:.2f}, expanded={result.expanded}, "
            f"min_clearance={meta.get('min_clearance', float('nan')):.2f} m"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- 命令
def cmd_run(args) -> int:
    cfg = build_scenario_config(args)
    scenario = registry.build(args.map, cfg)
    plan, timings = run_plan(scenario, args.algorithm, args, cfg.planner)
    report = evaluate_plan(plan, scenario.goals, scenario.world,
                           hard_clearance=cfg.planner.hard_clearance,
                           minimum_separation=cfg.planner.minimum_separation)

    if args.json:
        payload = {"scenario": scenario.info, "timings": timings, "metrics": report}
        print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
    else:
        print(describe(scenario, plan, report, timings, args))

    if args.plot:
        from .visualization import plot_scenario
        plot_scenario(scenario.world, plan.paths, scenario.starts, scenario.goals)
    return 0


def cmd_step(args) -> int:
    """A* 单步交互视图：一步一步看搜索过程。"""
    cfg = build_scenario_config(args)
    scenario = registry.build(args.map, cfg)

    from .visualization import interactive_astar
    start = scenario.world.world_to_grid(scenario.starts[0])
    goal = scenario.world.world_to_grid(scenario.goals[0])

    interactive_astar(
        scenario.world, start, goal, cfg.planner.to_weights(),
        heuristic=args.heuristic,
        diagonal=not args.no_diagonal,
        safety_margin=args.margin,
    )
    return 0


def cmd_ablation(args) -> int:
    """消融实验（对应论文 Sec. 4.5）。"""
    from .ablation import format_ablation, run_ablation

    cfg = build_scenario_config(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    report = run_ablation(cfg, map_name=args.map, seeds=seeds)

    text = format_ablation(report)
    print(text)
    if args.out:
        io.open(args.out, "w", encoding="utf-8").write(text)
        print()
        print(f"已写入 {args.out}")
    return 0


def cmd_info(args) -> int:
    """列出所有已注册的地图和算法。"""
    print("可用地图:")
    for name, label in registry.map_labels().items():
        print(f"  {name:16s} {label}")
    print()
    print("可用算法:")
    for name, label in algo_registry.algo_labels().items():
        print(f"  {name:16s} {label}")
    return 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UAV 3D Planner 模板 —— 可扩展的三维路径规划",
        epilog="加地图看 uav_planner/maps/，加算法看 uav_planner/algos/",
    )

    p.add_argument("--map", default="walls", help="地图名（--info 可列出全部）")
    p.add_argument("--algorithm", default="astar", help="算法名（--info 可列出全部）")
    p.add_argument("--seed", type=int, default=7, help="随机种子")
    p.add_argument("--uavs", type=int, default=3, help="无人机数量")
    p.add_argument("--obstacles", type=int, default=0,
                   help="额外随机障碍数（0 = 只有地图本身的结构）")

    g = p.add_argument_group("地图尺寸")
    g.add_argument("--resolution", type=float, default=1.0, help="每格米数")
    g.add_argument("--shape", type=int, nargs=3, default=(32, 32, 14),
                   metavar=("NX", "NY", "NZ"), help="网格尺寸")

    g = p.add_argument_group("约束与代价")
    g.add_argument("--hard", type=float, default=0.35, help="硬净空约束（米）")
    g.add_argument("--clearance", type=float, default=2.0, help="软净空半径（米）")
    g.add_argument("--min-separation", type=float, default=2.5, help="多机最小间距")
    g.add_argument("--margin", type=float, default=None,
                   help="passable 掩码额外膨胀（默认半个体素对角线）")
    g.add_argument("--max-expansions", type=int, default=250_000, help="扩展上限")
    g.add_argument("--heuristic", choices=("euclidean", "zero"), default="euclidean",
                   help="启发函数：euclidean=A*，zero=Dijkstra")
    g.add_argument("--no-diagonal", action="store_true", help="退化为 6 邻域")
    g.add_argument("--no-smooth", action="store_true", help="跳过平滑")

    p.add_argument("--step", action="store_true",
                   help="A* 单步交互视图（空格下一步 / a 跑到底 / h 切启发式）")
    p.add_argument("--dashboard", action="store_true",
                   help="打开浏览器驾驶舱（默认入口）")
    p.add_argument("--port", type=int, default=8899,
                   help="驾驶舱端口（默认 8899，避开常见占用）")
    p.add_argument("--no-browser", action="store_true",
                   help="不自动打开浏览器")
    p.add_argument("--plot", action="store_true", help="画出 3D 路径图")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.add_argument("--ablation", action="store_true",
                   help="消融实验：对比 A* 各组件贡献（论文 Sec. 4.5）")
    p.add_argument("--seeds", default="7", help="消融实验的随机种子，逗号分隔")
    p.add_argument("--out", default=None, help="把消融结果写入文件")
    p.add_argument("--info", action="store_true", help="列出所有地图和算法")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.ablation:
        return cmd_ablation(args)
    if args.info:
        return cmd_info(args)
    if args.dashboard:
        from .dashboard import serve
        return serve(port=args.port, open_browser=not args.no_browser)
    if args.step:
        return cmd_step(args)

    # 校验名字，给出友好错误
    if not registry.has_map(args.map):
        raise SystemExit(f"未知地图 {args.map!r}；可用: {list(registry.list_maps())}")
    if not algo_registry.has_algo(args.algorithm):
        raise SystemExit(
            f"未知算法 {args.algorithm!r}；可用: {list(algo_registry.list_algos())}")

    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
