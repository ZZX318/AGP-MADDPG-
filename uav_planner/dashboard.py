
"""浏览器驾驶舱 —— 本地 HTTP 服务。

接口
----
GET /              返回驾驶舱页面（单文件，JS 内联）
GET /api/meta      返回可用地图/算法（前端动态生成下拉框）
GET /api/plan      完整规划：路径 + 轨迹 + 双层指标
GET /api/step      单步搜索：把 A* 每一步录成帧

地图和算法都从注册表动态读取，加插件不用改这个文件。
"""

from __future__ import annotations

import json
import socket
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import algo_registry, registry
from .config import PlannerConfig, ScenarioConfig
from .dynamics import KinematicModel
from .fleet import FleetPlan, FleetPlanner
from .geometry import Box, Cylinder, Sphere
from .metrics import evaluate_plan

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "dashboard.html"

# 页面是单文件（HTML + CSS + JS 全内联），启动时读一次即可。
HTML_BODY = HTML_PATH.read_bytes()

PLAN_LOCK = threading.Lock()
PLAN_RUN_LOCK = threading.Lock()
PLAN_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
PLAN_CACHE_LIMIT = 16


def _args(query: dict) -> SimpleNamespace:
    """HTTP query -> 和 CLI 共享的参数形状。"""
    def value(name, default, cast=str):
        raw = query.get(name, [default])[0]
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return cast(default)

    shape_text = value("shape", "32x32x14").lower().replace("×", "x")
    try:
        shape = tuple(int(v) for v in shape_text.split("x"))
        if len(shape) != 3:
            raise ValueError
    except ValueError:
        shape = (32, 32, 14)

    return SimpleNamespace(
        map=value("map", "walls"),
        algorithm=value("algorithm", "astar"),
        seed=value("seed", 7, int),
        uavs=max(1, value("uavs", 1, int)),
        obstacles=max(0, value("obstacles", 0, int)),
        resolution=value("resolution", 1.0, float),
        shape=shape,
        hard=value("hard", 0.35, float),
        clearance=value("clearance", 2.0, float),
        min_separation=value("min_separation", 2.5, float),
        margin=None,
        max_expansions=value("max_expansions", 250_000, int),
        heuristic=value("heuristic", "euclidean"),
        no_diagonal=value("diagonal", "1") in ("0", "false", "no"),
        no_smooth=value("smooth", "1") in ("0", "false", "no"),
    )


def _planner_config(args) -> PlannerConfig:
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


def _scenario_config(args, n_uavs=None) -> ScenarioConfig:
    return ScenarioConfig(
        seed=args.seed,
        n_uavs=args.uavs if n_uavs is None else n_uavs,
        n_obstacles=args.obstacles,
        planner=_planner_config(args),
    )


def _safe(value):
    """递归转成 JSON 安全类型（inf -> None）。"""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _no_fly_boxes(scenario) -> list:
    """返回禁飞结构对应的 box 列表（用于从障碍里排除，避免画两遍）。"""
    info = scenario.info or {}
    world = scenario.world
    res = world.resolution
    origin = world.origin
    shape = world.shape
    boxes = []

    top = info.get("no_fly_top")
    if top:
        z_fly_max = info.get("z_fly_max", shape[2] - 1 - top)
        boxes.append((
            (float(origin[0]), float(origin[1]), float(origin[2] + (z_fly_max + 1) * res)),
            (float(origin[0] + (shape[0] - 1) * res),
             float(origin[1] + (shape[1] - 1) * res),
             float(origin[2] + (shape[2] - 1) * res)),
        ))

    for b in info.get("no_fly_bands", []):
        boxes.append((
            (float(origin[0] + b["x_lo"] * res),
             float(origin[1] + b["y_lo"] * res),
             float(origin[2] + b["z_lo"] * res)),
            (float(origin[0] + b["x_hi"] * res),
             float(origin[1] + b["y_hi"] * res),
             float(origin[2] + b["z_hi"] * res)),
        ))
    return boxes


def _is_no_fly(obstacle, no_fly_boxes, tol: float = 1e-6) -> bool:
    """判断这个 box 是不是禁飞结构（坐标完全一致）。"""
    try:
        lo = tuple(float(v) for v in obstacle.minimum)
        hi = tuple(float(v) for v in obstacle.maximum)
    except AttributeError:
        return False
    for (nlo, nhi) in no_fly_boxes:
        if all(abs(lo[i] - nlo[i]) < tol for i in range(3)) and \
           all(abs(hi[i] - nhi[i]) < tol for i in range(3)):
            return True
    return False


def _obstacle_geometry(scenario) -> dict:
    """点云 + 解析几何。前端优先用解析几何画真形状。

    注意：禁飞结构（顶部禁飞层 / 横向禁飞带）**不**放进 geometry，
    它们由 no_fly 字段单独提供、用半透明方式渲染。
    否则前端会把同一个盒子画两遍（一次实心、一次玻璃），实心那次会挡住。
    """
    occupied = np.argwhere(scenario.world.occupied)
    stride = max(1, int(np.ceil(len(occupied) / 4500)))
    points = [scenario.world.grid_to_world(tuple(c)).tolist()
              for c in occupied[::stride]]

    no_fly_boxes = _no_fly_boxes(scenario)

    geometry = []
    for obstacle in scenario.world.obstacles:
        if isinstance(obstacle, Sphere):
            geometry.append({"type": "sphere",
                             "center": obstacle.center.tolist(),
                             "radius": float(obstacle.radius)})
        elif isinstance(obstacle, Box):
            # 禁飞结构交给 no_fly 字段，这里跳过
            if _is_no_fly(obstacle, no_fly_boxes):
                continue
            geometry.append({"type": "box",
                             "minimum": obstacle.minimum.tolist(),
                             "maximum": obstacle.maximum.tolist()})
        elif isinstance(obstacle, Cylinder):
            geometry.append({"type": "cylinder",
                             "center_xy": obstacle.center_xy.tolist(),
                             "z_min": float(obstacle.z_min),
                             "z_max": float(obstacle.z_max),
                             "radius": float(obstacle.radius)})
    return {"points": points, "geometry": geometry}


def _no_fly_zones(scenario) -> dict:
    """从 scenario.info 提取禁飞区描述，供前端半透明渲染。

    地图插件可以往 info 里写 no_fly_bands / no_fly_top / z_fly_max，
    这里统一转成前端能直接画的结构。
    """
    info = scenario.info or {}
    world = scenario.world
    res = world.resolution
    origin = world.origin
    shape = world.shape

    zones = []

    # 顶部禁飞层：铺满 x-y
    top = info.get("no_fly_top")
    if top:
        z_fly_max = info.get("z_fly_max", shape[2] - 1 - top)
        zones.append({
            "kind": "top",
            "minimum": [float(origin[0]), float(origin[1]), float(origin[2] + (z_fly_max + 1) * res)],
            "maximum": [float(origin[0] + (shape[0] - 1) * res),
                        float(origin[1] + (shape[1] - 1) * res),
                        float(origin[2] + (shape[2] - 1) * res)],
        })

    # 横向禁飞带
    for b in info.get("no_fly_bands", []):
        zones.append({
            "kind": "band",
            "minimum": [float(origin[0] + b["x_lo"] * res),
                        float(origin[1] + b["y_lo"] * res),
                        float(origin[2] + b["z_lo"] * res)],
            "maximum": [float(origin[0] + b["x_hi"] * res),
                        float(origin[1] + b["y_hi"] * res),
                        float(origin[2] + b["z_hi"] * res)],
        })

    return {
        "zones": zones,
        "z_fly_max": info.get("z_fly_max"),
        "gap": {
            "wall_x": info.get("wall_x"),
            "gap_y": info.get("gap_y"),
            "gap_z": info.get("gap_z"),
            "gap_w": info.get("gap_w"),
            "gap_h": info.get("gap_h"),
        } if info.get("wall_x") is not None else None,
    }


# ------------------------------------------------------------------ /api/plan
def plan_payload(query: dict) -> dict:
    args = _args(query)
    if not registry.has_map(args.map):
        args.map = "walls"
    if not algo_registry.has_algo(args.algorithm):
        args.algorithm = "astar"

    key = tuple(sorted((k, tuple(v)) for k, v in query.items()))
    with PLAN_LOCK:
        cached = PLAN_CACHE.get(key)
        if cached is not None:
            PLAN_CACHE.move_to_end(key)
            return cached

    with PLAN_RUN_LOCK:
        return _plan_uncached(args, key)


def _plan_uncached(args, key) -> dict:
    cfg = _scenario_config(args)
    scenario = registry.build(args.map, cfg)
    world = scenario.world
    weights = cfg.planner.to_weights()

    starts = [world.world_to_grid(p) for p in scenario.starts]
    goals = [world.world_to_grid(p) for p in scenario.goals]

    t0 = time.perf_counter()
    results = algo_registry.plan_fleet(
        args.algorithm, world, starts, goals, weights,
        diagonal=not args.no_diagonal,
        max_expansions=args.max_expansions,
        heuristic=args.heuristic,
        safety_margin=args.margin,
        seed=args.seed,
    )
    planning_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    fleet = FleetPlanner(world, weights)
    paths = [fleet._post_process(r.points, smooth=not args.no_smooth)
             for r in results]
    plan = FleetPlan(paths=paths, results=results)
    fleet.rollout(plan, scenario.goals, KinematicModel(),
                  hard_clearance=weights.hard_clearance)
    rollout_time = time.perf_counter() - t1

    timings = {"planning": planning_time, "rollout": rollout_time}

    report = evaluate_plan(plan, scenario.goals, world,
                           hard_clearance=weights.hard_clearance,
                           minimum_separation=weights.minimum_separation)

    obstacle_view = _obstacle_geometry(scenario)
    payload = {
        "mode": "flight",
        "scenario": scenario.info,
        "no_fly": _no_fly_zones(scenario),
        "bounds": {"origin": world.origin.tolist(),
                   "maximum": world.maximum.tolist(),
                   "resolution": world.resolution},
        "obstacles": obstacle_view["points"],
        "obstacles_geometry": obstacle_view["geometry"],
        "starts": np.asarray(scenario.starts).tolist(),
        "goals": np.asarray(scenario.goals).tolist(),
        "paths": [np.asarray(p).tolist() for p in plan.paths],
        "trajectories": [np.asarray(p).tolist() for p in plan.trajectories],
        "metrics": report,
        "timings": timings,
        "results": [{"planner": r.planner, "cost": r.cost,
                     "expanded": r.expanded, "metadata": r.metadata}
                    for r in plan.results],
        "controls": {"map": args.map, "algorithm": args.algorithm,
                     "seed": args.seed, "uavs": args.uavs,
                     "obstacles": args.obstacles},
    }
    payload = _safe(payload)

    with PLAN_LOCK:
        PLAN_CACHE[key] = payload
        PLAN_CACHE.move_to_end(key)
        while len(PLAN_CACHE) > PLAN_CACHE_LIMIT:
            PLAN_CACHE.popitem(last=False)
    return payload


# ------------------------------------------------------------------ /api/step
def _variant_flags(algorithm: str):
    """把算法名映射到 (hard, obs, sep) 三个消融开关。"""
    mapping = {
        "astar": (True, True, True),
        "astar-standard": (False, False, False),
        "astar-hard": (True, False, False),
        "astar-hard-obs": (True, True, False),
    }
    return mapping.get(algorithm, (True, True, True))


def step_payload(query: dict) -> dict:
    """把 A* 的搜索过程录成帧，供前端单步播放。"""
    args = _args(query)
    if not registry.has_map(args.map):
        args.map = "walls"

    cfg = _scenario_config(args, n_uavs=1)
    scenario = registry.build(args.map, cfg)
    world = scenario.world
    weights = cfg.planner.to_weights()

    start = world.world_to_grid(scenario.starts[0])
    goal = world.world_to_grid(scenario.goals[0])

    # 根据所选算法决定三个消融开关，让搜索模式也能看到变体差异
    flags = _variant_flags(args.algorithm)

    from .astar3d import AStarStepper3D
    stepper = AStarStepper3D(
        world, start, goal, weights,
        heuristic=args.heuristic,
        diagonal=not args.no_diagonal,
        max_expansions=args.max_expansions,
        safety_margin=args.margin,
        use_hard_constraint=flags[0],
        use_obstacle_cost=flags[1],
        use_separation_cost=flags[2],
    )

    # 先跑一遍拿到总扩展数，据此决定每帧合并多少步
    stepper.run()
    total_expanded = int(stepper.expanded)
    reached = bool(stepper.reached)
    stride = max(1, int(np.ceil(total_expanded / 400)))

    # 重跑一遍，记录每一帧
    stepper.reset()
    frames = []
    while not stepper.done:
        close_nodes, open_nodes = [], []
        for _ in range(stride):
            before_open = set(stepper.open_set)
            stepper.step()
            if stepper.current is not None:
                close_nodes.append(
                    world.grid_to_world(stepper.table.cell_of(stepper.current)).tolist())
            for idx in stepper.open_set - before_open:
                info = stepper.node_info(idx)
                pos = world.grid_to_world(stepper.table.cell_of(idx)).tolist()
                open_nodes.append([pos[0], pos[1], pos[2], round(info["f"], 2)])
            if stepper.done:
                break

        curinfo = None
        if stepper.current is not None:
            info = stepper.node_info(stepper.current)
            curinfo = {"g": round(info["g"], 2), "h": round(info["h"], 2),
                       "f": round(info["f"], 2)}
        frames.append({
            "close": close_nodes,
            "open": open_nodes,
            "exp": int(stepper.expanded),
            "osize": len(stepper.open_set),
            "cur": close_nodes[-1] if close_nodes else None,
            "curinfo": curinfo,
        })

    obstacle_view = _obstacle_geometry(scenario)
    payload = {
        "mode": "step",
        "no_fly": _no_fly_zones(scenario),
        "bounds": {"origin": world.origin.tolist(),
                   "maximum": world.maximum.tolist(),
                   "resolution": world.resolution},
        "obstacles": obstacle_view["points"],
        "obstacles_geometry": obstacle_view["geometry"],
        "start": np.asarray(scenario.starts[0]).tolist(),
        "goal": np.asarray(scenario.goals[0]).tolist(),
        "frames": frames,
        "path": [np.asarray(p).tolist() for p in stepper.path],
        "total_expanded": total_expanded,
        "reached": reached,
        "stride": stride,
        "heuristic": args.heuristic,
    }
    return _safe(payload)


# ------------------------------------------------------------------ /api/meta
def ablation_payload(query: dict) -> dict:
    """消融实验：对比改进 A* 各组件贡献（论文 Sec. 4.5）。"""
    from .ablation import DEFAULT_VARIANTS, run_ablation

    args = _args(query)
    cfg = _scenario_config(args)

    # 支持前端传自定义变体组合，格式 "hard,obs,sep"（缺省表示关闭）
    raw = query.get("variants", [""])[0].strip()
    variants = None
    if raw:
        variants = []
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(",")]
            name = parts[0]
            flags = set(parts[1:])
            variants.append(type(DEFAULT_VARIANTS[0])(
                name=name,
                description=name,
                use_hard_constraint="hard" in flags,
                use_obstacle_cost="obs" in flags,
                use_separation_cost="sep" in flags,
            ))

    seeds_raw = query.get("seeds", [str(args.seed)])[0]
    try:
        seeds = [int(s) for s in seeds_raw.split(",") if s.strip()]
    except ValueError:
        seeds = [args.seed]
    if not seeds:
        seeds = [args.seed]

    report = run_ablation(cfg, variants=variants, map_name=args.map, seeds=seeds)
    report["mode"] = "ablation"
    return _safe(report)


def meta_payload() -> dict:
    return {
        "maps": [{"name": n, "label": registry.map_labels()[n]}
                 for n in registry.list_maps()],
        "algorithms": [{"name": n, "label": algo_registry.algo_labels()[n]}
                       for n in algo_registry.list_algos()],
        "shapes": [[24, 24, 12], [32, 32, 14], [40, 40, 16]],
    }


# -------------------------------------------------------------------- server
class _Server(ThreadingHTTPServer):
    """关掉 SO_REUSEADDR，让端口占用能被正确检测到。"""

    allow_reuse_address = False
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    server_version = "UAVPlannerCockpit/1.0"

    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML_BODY)
            return

        routes = {
            "/api/meta": lambda: meta_payload(),
            "/api/plan": lambda: plan_payload(parse_qs(parsed.query)),
            "/api/step": lambda: step_payload(parse_qs(parsed.query)),
            "/api/ablation": lambda: ablation_payload(parse_qs(parsed.query)),
        }
        handler = routes.get(parsed.path)
        if handler is None:
            self._send(404, "text/plain; charset=utf-8", b"Not Found")
            return

        try:
            body = json.dumps(handler(), ensure_ascii=False).encode()
            self._send(200, "application/json; charset=utf-8", body)
        except Exception as exc:  # noqa: BLE001
            body = json.dumps(
                {"error": f"{type(exc).__name__}: {exc}",
                 "hint": "试试降低障碍数、换小一点的地图，或调大硬净空容差"},
                ensure_ascii=False).encode()
            self._send(400, "application/json; charset=utf-8", body)

    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8899, open_browser: bool = True) -> int:
    """启动驾驶舱。

    如果指定端口被占用（很常见：另一个实验台、豆包之类的工具），
    自动往后找一个空闲端口，而不是直接崩掉。
    """
    httpd = None
    last_error = None
    chosen = None

    for candidate in range(int(port), int(port) + 20):
        # 先显式探测端口是否空闲。
        # 注意：Windows 上 HTTPServer 的 SO_REUSEADDR 语义与 Unix 不同，
        # 直接构造 ThreadingHTTPServer 时端口被占用也可能"绑定成功"，
        # 所以必须先自己 bind 一次确认。
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 不要设 SO_REUSEADDR：Windows 上它会让"已被占用"的端口也能 bind 成功，
        # 探测就失去意义了。
        try:
            probe.bind((host, candidate))
            probe.close()
            chosen = candidate
        except OSError as exc:
            last_error = exc
            probe.close()
            continue

        try:
            httpd = _Server((host, candidate), _Handler)
            break
        except OSError as exc:
            last_error = exc
            httpd = None
            continue

    if httpd is None:
        print(f"无法在 {port}~{port + 19} 之间找到空闲端口: {last_error}")
        return 1

    url = f"http://{host}:{httpd.server_port}/"
    if httpd.server_port != int(port):
        print(f"端口 {port} 已被占用，自动改用 {httpd.server_port}")
    print(f"UAV 实验驾驶舱: {url}")
    print("关闭页面后在终端按 Ctrl+C 结束服务。")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
