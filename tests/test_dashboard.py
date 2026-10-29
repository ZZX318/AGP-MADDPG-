
"""驾驶舱后端测试 —— 不启服务器，直接调 payload 函数。"""

from __future__ import annotations

import sys
import pathlib
import json

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_planner.dashboard import meta_payload, plan_payload, step_payload


def test_meta_lists_registry_contents():
    """meta 必须反映注册表内容，前端靠它动态生成下拉框。"""
    m = meta_payload()
    assert any(x["name"] == "walls" for x in m["maps"]), "地图列表缺 walls"
    names = {x["name"] for x in m["algorithms"]}
    # 改进 A* + 三个消融变体
    expected = {"astar", "astar-standard", "astar-hard", "astar-hard-obs"}
    assert names == expected, f"算法列表应为 {expected}，实际: {names}"
    assert len(m["shapes"]) == 3
    print(f"  地图 {[x['name'] for x in m['maps']]}")
    print(f"  算法 {sorted(names)}")


def test_plan_payload_shape():
    """plan payload 必须含前端需要的全部字段。"""
    p = plan_payload({"map": ["walls"], "algorithm": ["astar"], "seed": ["7"],
                      "uavs": ["1"], "obstacles": ["0"], "hard": ["0.35"]})
    for key in ("mode", "bounds", "obstacles", "obstacles_geometry",
                "starts", "goals", "paths", "trajectories",
                "metrics", "results", "controls"):
        assert key in p, f"缺字段 {key}"

    assert p["mode"] == "flight"
    assert len(p["paths"]) == 1
    assert len(p["trajectories"]) == 1
    assert len(p["obstacles_geometry"]) > 0, "缺口墙应该有 4 个 box"
    assert "plan" in p["metrics"] and "execution" in p["metrics"]
    print(f"  路径 {len(p['paths'])}, 轨迹 {len(p['trajectories'])}, "
          f"几何 {len(p['obstacles_geometry'])}")


def test_plan_respects_clearance():
    """plan payload 报的净空必须守住硬约束。"""
    p = plan_payload({"map": ["walls"], "algorithm": ["astar"], "seed": ["7"],
                      "uavs": ["1"], "obstacles": ["0"], "hard": ["0.35"]})
    layer = p["metrics"]["plan"]
    assert not layer["clearance_violation"], "净空违规"
    assert layer["min_clearance"] >= 0.35 - 1e-9
    print(f"  净空 {layer['min_clearance']:.2f} m >= 0.35 m")


def test_plan_is_json_serializable():
    """payload 必须能直接 json.dumps（inf 要转成 None）。"""
    p = plan_payload({"map": ["walls"], "algorithm": ["astar"], "seed": ["7"],
                      "uavs": ["1"], "obstacles": ["0"], "hard": ["0.35"]})
    text = json.dumps(p, ensure_ascii=False)
    assert "Infinity" not in text and "NaN" not in text
    print(f"  JSON {len(text)} 字节，无 Infinity/NaN")


def test_plan_cache_hits():
    """相同 query 第二次应该命中缓存（返回同一对象）。"""
    q = {"map": ["walls"], "algorithm": ["astar"], "seed": ["13"],
         "uavs": ["1"], "obstacles": ["0"], "hard": ["0.35"]}
    a = plan_payload(q)
    b = plan_payload(q)
    assert a is b, "缓存没命中"
    print("  缓存命中 OK")


def test_step_payload_shape():
    """step payload 必须含单步播放需要的字段。"""
    s = step_payload({"map": ["walls"], "seed": ["7"],
                      "heuristic": ["euclidean"], "hard": ["0.35"]})
    for key in ("mode", "bounds", "obstacles_geometry", "start", "goal",
                "frames", "path", "total_expanded", "reached", "heuristic"):
        assert key in s, f"缺字段 {key}"

    assert s["mode"] == "step"
    assert s["reached"] is True
    assert len(s["frames"]) > 0
    assert len(s["path"]) >= 2
    # 每帧结构
    f = s["frames"][0]
    for k in ("close", "open", "exp", "osize", "cur", "curinfo"):
        assert k in f, f"帧缺字段 {k}"
    print(f"  {len(s['frames'])} 帧, 总扩展 {s['total_expanded']}, "
          f"路径 {len(s['path'])} 点")


def test_step_heuristic_changes_expansion():
    """Dijkstra 的扩展数应该显著多于 A*。"""
    base = {"map": ["walls"], "seed": ["7"], "hard": ["0.35"]}
    a = step_payload({**base, "heuristic": ["euclidean"]})
    d = step_payload({**base, "heuristic": ["zero"]})
    assert d["total_expanded"] > a["total_expanded"], (
        f"Dijkstra {d['total_expanded']} 应 > A* {a['total_expanded']}")
    print(f"  A* {a['total_expanded']} vs Dijkstra {d['total_expanded']}")


def test_unknown_controls_fallback():
    """未知地图/算法要回退到默认值，不能报错。"""
    p = plan_payload({"map": ["nonexistent"], "algorithm": ["nonexistent"],
                      "seed": ["7"], "uavs": ["1"]})
    assert p["controls"]["map"] == "walls"
    assert p["controls"]["algorithm"] == "astar"
    print("  未知名字回退 OK")


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
