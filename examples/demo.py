"""最小示例：不用命令行，直接在代码里调用。

运行：python examples/demo.py
"""

import sys
import pathlib

# 让脚本能直接跑，不用先 pip install
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_planner import algos, make_scenario, maps, plan_fleet
from uav_planner.config import PlannerConfig, ScenarioConfig
from uav_planner.fleet import FleetPlanner
from uav_planner.metrics import evaluate_plan


def main():
    # 1. 看看有哪些地图和算法
    print("地图:", maps())
    print("算法:", algos())

    # 2. 构造场景
    cfg = ScenarioConfig(
        seed=7, n_uavs=2, n_obstacles=0,
        planner=PlannerConfig(shape=(32, 32, 14), resolution=1.0, hard_clearance=0.35),
    )
    scenario = make_scenario("walls", cfg)
    print(f"障碍 {scenario.info['obstacles']} 个, "
          f"占用率 {scenario.info['occupancy']:.1%}")

    # 3. 规划（只算路径，不算执行轨迹）
    starts = [scenario.world.world_to_grid(p) for p in scenario.starts]
    goals = [scenario.world.world_to_grid(p) for p in scenario.goals]
    results = plan_fleet("astar", scenario.world, starts, goals,
                         cfg.planner.to_weights())
    for i, r in enumerate(results):
        print(f"UAV {i}: {len(r.points)} 航点, cost={r.cost:.2f}, "
              f"expanded={r.expanded}")

    # 4. 完整流程：规划 -> 平滑 -> 动力学 rollout
    fleet = FleetPlanner.from_config(scenario.world, cfg.planner)
    plan = fleet.plan(starts, goals)          # 含平滑
    fleet.rollout(plan, scenario.goals,        # 生成执行轨迹
                  hard_clearance=cfg.planner.hard_clearance)

    # 5. 双份指标
    report = evaluate_plan(plan, scenario.goals, scenario.world,
                           hard_clearance=cfg.planner.hard_clearance)
    print()
    print(f"规划层净空: {report['plan']['min_clearance']:.2f} m")
    print(f"执行层净空: {report['execution']['min_clearance']:.2f} m")
    print(f"硬约束 {cfg.planner.hard_clearance} m -> "
          f"{'OK' if not report['execution']['clearance_violation'] else 'VIOLATED'}")


if __name__ == "__main__":
    main()
