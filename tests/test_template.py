
"""模板工程的自检测试。

不依赖 pytest，直接 python tests/test_template.py 就能跑。
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_planner import algo_registry, registry, make_scenario, plan_fleet
from uav_planner.config import PlannerConfig, ScenarioConfig
from uav_planner.metrics import evaluate_plan
from uav_planner.fleet import FleetPlanner


def _cfg(**kwargs):
    defaults = dict(
        seed=7, n_uavs=2, n_obstacles=0,
        planner=PlannerConfig(shape=(32, 32, 14), resolution=1.0, hard_clearance=0.35),
    )
    defaults.update(kwargs)
    return ScenarioConfig(**defaults)


def test_registries_populated():
    """两个注册表都应该有内容。"""
    assert "walls" in registry.list_maps(), f"地图未注册: {registry.list_maps()}"
    assert "astar" in algo_registry.list_algos(), f"算法未注册: {algo_registry.list_algos()}"
    # 改进 A* + 三个消融变体，共 4 个
    expected = {"astar", "astar-standard", "astar-hard", "astar-hard-obs"}
    actual = set(algo_registry.list_algos())
    assert actual == expected, f"算法应为 {expected}，实际: {actual}"
    print("  地图:", registry.list_maps())
    print("  算法:", algo_registry.list_algos())


def test_map_builds():
    """地图能构造，且障碍、端点都正常。"""
    sc = make_scenario("walls", _cfg())
    assert sc.world.occupied.any(), "地图没有任何障碍"
    assert len(sc.starts) == 2 and len(sc.goals) == 2
    assert sc.info["layout"] == "walls"
    print(f"  障碍 {sc.info['obstacles']} 个, 占用率 {sc.info['occupancy']:.1%}")


def test_endpoints_are_safe_and_connected():
    """端点必须在可通行掩码内，且与终点连通。"""
    cfg = _cfg()
    sc = make_scenario("walls", cfg)
    hard = cfg.planner.hard_clearance
    margin = 0.5 * np.sqrt(3.0) * cfg.planner.resolution

    mask = sc.world.passable(hard, margin)
    for i, p in enumerate(sc.starts + sc.goals):
        assert mask[sc.world.world_to_grid(p)], f"端点 {i} 落在障碍里"

    for s, g in zip(sc.starts, sc.goals):
        assert sc.world.reachable(s, g, hard, margin), "起终点不连通"
    print("  端点净空 + 连通性 OK")


def test_astar_plans_and_respects_clearance():
    """A* 能规划，且净空守住硬约束。"""
    cfg = _cfg()
    sc = make_scenario("walls", cfg)
    hard = cfg.planner.hard_clearance

    starts = [sc.world.world_to_grid(p) for p in sc.starts]
    goals = [sc.world.world_to_grid(p) for p in sc.goals]
    results = algo_registry.plan_fleet("astar", sc.world, starts, goals,
                                       cfg.planner.to_weights())
    assert len(results) == 2
    assert all(len(r.points) >= 2 for r in results)

    plan = FleetPlanner.from_config(sc.world, cfg.planner).plan(starts, goals)
    report = evaluate_plan(plan, sc.goals, sc.world, hard_clearance=hard)["plan"]
    assert not report["clearance_violation"], "净空违规"
    print(f"  净空 {report['min_clearance']:.2f} m >= {hard} m, "
          f"路径 {report['mean_path_length']:.1f} m")


def test_same_seed_is_deterministic():
    """同 seed 必须产生完全一样的地图。"""
    a = make_scenario("walls", _cfg(seed=11))
    b = make_scenario("walls", _cfg(seed=11))
    assert np.array_equal(a.world.occupied, b.world.occupied)
    assert np.allclose(a.starts, b.starts)
    assert np.allclose(a.goals, b.goals)
    print("  同 seed 位一致 OK")


def test_unknown_names_raise():
    """未知地图/算法要给出清晰错误。"""
    try:
        registry.get_map("nonexistent")
    except KeyError as exc:
        assert "available" in str(exc)
    else:
        raise AssertionError("未知地图应该抛 KeyError")

    try:
        algo_registry.get_algo("nonexistent")
    except KeyError as exc:
        assert "available" in str(exc)
    else:
        raise AssertionError("未知算法应该抛 KeyError")
    print("  未知名字报错 OK")


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
