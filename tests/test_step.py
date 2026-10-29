
"""单步视图测试 —— 不弹窗口，直接驱动状态机验证数值正确性。"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_planner.astar3d import AStar3D, AStarStepper3D
from uav_planner.config import PlannerConfig, ScenarioConfig
from uav_planner import make_scenario


def _setup(seed=7):
    cfg = ScenarioConfig(
        seed=seed, n_uavs=1, n_obstacles=0,
        planner=PlannerConfig(shape=(32, 32, 14), resolution=1.0, hard_clearance=0.35),
    )
    sc = make_scenario("walls", cfg)
    world = sc.world
    start = world.world_to_grid(sc.starts[0])
    goal = world.world_to_grid(sc.goals[0])
    return world, start, goal, cfg


def test_stepper_reaches_goal():
    """单步状态机必须能跑到终点。"""
    world, start, goal, cfg = _setup()
    stepper = AStarStepper3D(world, start, goal, cfg.planner.to_weights())
    stepper.run()
    assert stepper.done and stepper.reached, "单步没到达终点"
    assert len(stepper.path) >= 2
    print(f"  扩展 {stepper.expanded} 个节点，路径 {len(stepper.path)} 点")


def test_stepper_matches_planner():
    """单步的扩展数必须和真实 A* 一致（这是它可信的前提）。"""
    world, start, goal, cfg = _setup()
    weights = cfg.planner.to_weights()

    stepper = AStarStepper3D(world, start, goal, weights)
    stepper.run()

    planner = AStar3D(world, weights)
    result = planner.plan(start, goal)

    assert stepper.expanded == result.expanded, (
        f"单步扩展 {stepper.expanded} != 规划器 {result.expanded}")
    assert np.allclose(stepper.path, result.points), "路径不一致"
    print(f"  扩展数一致（{stepper.expanded}），路径一致")


def test_stepper_step_by_step():
    """手动一步一步 step()，状态应该逐步推进。"""
    world, start, goal, cfg = _setup()
    stepper = AStarStepper3D(world, start, goal, cfg.planner.to_weights())

    assert stepper.expanded == 0
    assert stepper.current is None

    stepper.step()
    assert stepper.expanded == 1
    assert stepper.current is not None
    first_current = stepper.current

    info = stepper.node_info(first_current)
    assert np.isfinite(info["g"]) and np.isfinite(info["f"])
    assert info["f"] >= info["g"] - 1e-9      # f = g + h >= g
    print(f"  第 1 步: g={info['g']:.2f} h={info['h']:.2f} f={info['f']:.2f}")


def test_heuristic_toggle_changes_expansions():
    """A* 和 Dijkstra 的扩展数应该不同（h=0 通常扩展更多）。"""
    world, start, goal, cfg = _setup()
    weights = cfg.planner.to_weights()

    astar = AStarStepper3D(world, start, goal, weights, heuristic="euclidean")
    astar.run()

    dijkstra = AStarStepper3D(world, start, goal, weights, heuristic="zero")
    dijkstra.run()

    assert astar.reached and dijkstra.reached
    assert dijkstra.expanded >= astar.expanded, (
        f"Dijkstra 扩展 {dijkstra.expanded} 应 >= A* {astar.expanded}")
    print(f"  A* {astar.expanded} 次扩展 vs Dijkstra {dijkstra.expanded} 次")


def test_reset_restores_initial_state():
    """reset() 应该把状态清干净。"""
    world, start, goal, cfg = _setup()
    stepper = AStarStepper3D(world, start, goal, cfg.planner.to_weights())
    for _ in range(5):
        stepper.step()
    assert stepper.expanded > 0

    stepper.reset()
    assert stepper.expanded == 0
    assert stepper.current is None
    assert not stepper.path
    assert len(stepper.open_set) == 1
    print("  reset 后状态干净")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"RUN  {fn.__name__}")
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
