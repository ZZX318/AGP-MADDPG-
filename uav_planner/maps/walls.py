
"""缺口墙 + 禁飞区地图。

设计意图
--------
这是一张**强制三维绕行**的基准图，四个结构各有分工：

1. **缺口墙**（竖直，横跨 y-z 截面）
   只有中间一个矩形洞。26 邻域无法在 xy 平面内绕过，
   路径必须用 z 轴。它检验"规划器是否真的会做三维绕行"。

2. **顶部禁飞层**（横向，铺满整个 x-y 平面）
   在 z 的上部封一层，防止无人机从墙顶"飞过去"走捷径。
   墙只到某个高度时，飞行器完全可以爬升越过 —— 加了禁飞层就不行。

3. **横向禁飞区**（沿 x 方向贯穿的条带，但只占部分高度）
   在 y 的某些区间立一道"矮墙"，从地面到中等高度。
   它不封顶，所以路径要么从上面越过，要么从旁边绕过 ——
   这检验规划器会不会做"垂直 + 水平"的组合绕行。
   如果把它做成全高，整张图会被切成互不连通的区域，那就不是规划问题了。

关键实现顺序
------------
**必须先建结构，再采端点**。
先采端点后加禁飞层，端点可能正好落在禁飞结构里，
随后 _finish() 会判定不连通并疯狂删障碍，把设计好的结构全删光。
"""

from __future__ import annotations

import numpy as np

from ..config import ScenarioConfig
from ..environment import VoxelMap
from ..registry import register_map
from ..scenario import Scenario, _finish, _sample_endpoints


@register_map("walls", "缺口墙 + 禁飞区")
def make_walls(config: ScenarioConfig | None = None) -> Scenario:
    """生成缺口墙 + 顶部禁飞层 + 横向禁飞区。

    参数
    ----
    config.seed          决定缺口位置、禁飞条带位置、起终点
    config.n_uavs        无人机数量
    config.n_obstacles   额外随机球体数量（0 = 只有结构障碍）
    config.planner       地图尺寸 / 分辨率 / 硬净空
    """
    config = config or ScenarioConfig()
    planner_cfg = config.planner

    world = VoxelMap(shape=planner_cfg.shape, resolution=planner_cfg.resolution)
    clearance = float(planner_cfg.hard_clearance)
    margin = (0.5 * np.sqrt(3.0) * planner_cfg.resolution
              if planner_cfg.safety_margin is None else float(planner_cfg.safety_margin))

    rng = np.random.default_rng(config.seed)
    nx, ny, nz = world.shape
    res = world.resolution

    # 顶部禁飞层厚度：至少 1 层，最多 nz 的 1/4（且不超过 3）
    top_band = max(1, min(nz // 4, 3))
    z_fly_max = nz - 1 - top_band        # 可飞行的最高 z 层

    # ================================================================
    # 第 1 步：先建所有结构（此时还没有端点，不会误埋）
    # ================================================================

    # ---- 1.1 缺口墙：横跨 y-z，中间一个洞 ----
    x_wall = int(round(nx * 0.5))
    gap_w = max(4, min(7, ny // 5))                  # 洞口 y 向宽度

    # 洞口尺寸必须给四周留出实体，否则补集会出现零厚度矩形。
    # z 方向：上下各至少 2 格实体 -> gap_h <= z_fly_max - 3
    gap_h = max(2, min(5, z_fly_max - 3))
    gap_h = max(2, gap_h)
    # 洞的 z 起点：至少 2，且洞顶 <= z_fly_max - 2
    z_lo_min, z_lo_max = 2, max(2, z_fly_max - gap_h - 2)
    gap_z = int(rng.integers(z_lo_min, z_lo_max + 1)) if z_lo_max > z_lo_min else z_lo_min
    gap_z = int(np.clip(gap_z, z_lo_min, z_lo_max))
    # y 方向：左右各至少 2 格实体
    gap_w = max(4, min(gap_w, ny - 5))
    y_lo_min, y_lo_max = 2, max(2, ny - gap_w - 3)
    gap_y = int(rng.integers(y_lo_min, y_lo_max + 1)) if y_lo_max > y_lo_min else y_lo_min
    gap_y = int(np.clip(gap_y, y_lo_min, y_lo_max))

    # ---- 用"整面墙减去一个洞"的方式构造 ----
    # 洞在 y 方向占 [gy0, gy1)，z 方向占 [gz0, gz1)。
    # 补集 = 4 块互不重叠的矩形，逐块检查非空后才添加。
    gy0, gy1 = gap_y, gap_y + gap_w
    gz0, gz1 = gap_z, gap_z + gap_h

    def add_slab(y0, y1, z0, z1):
        """只在矩形非空时加一块墙。坐标都是闭区间（含端点）。

        Box 要求 maximum 严格大于 minimum（三维都要），
        所以单格厚度也必须满足：y1 > y0 且 z1 > z0。
        """
        y0 = max(0, int(y0)); y1 = min(ny - 1, int(y1))
        z0 = max(0, int(z0)); z1 = min(z_fly_max, int(z1))
        if y1 <= y0 or z1 <= z0:
            return
        x0 = max(0, x_wall)
        x1 = min(nx - 1, x_wall + 1)
        if x1 <= x0:
            return
        world.add_box((x0, y0, z0), (x1, y1, z1))

    # 1) 洞下方整条
    add_slab(0, ny - 1, 0, gz0 - 1)
    # 2) 洞上方整条
    add_slab(0, ny - 1, gz1, z_fly_max)
    # 3) 洞左侧（z 在洞的高度范围内）
    add_slab(0, gy0 - 1, gz0, gz1 - 1)
    # 4) 洞右侧（z 在洞的高度范围内）
    add_slab(gy1, ny - 1, gz0, gz1 - 1)

    # ---- 1.2 顶部禁飞层：铺满整个 x-y ----
    world.add_box((0, 0, z_fly_max + 1), (nx - 1, ny - 1, nz - 1))

    # ---- 1.3 横向禁飞区：沿 x 贯穿，但只到中等高度 ----
    # 高度取可飞高度的 60%~75%，留出上方通道
    band_top = max(2, int(round(z_fly_max * 0.7)))
    n_bands = 2
    band_thick = max(2, ny // 10)
    band_regions = []

    for k in range(n_bands):
        # 两条带错开：一条在左半区，一条在右半区
        if k % 2 == 0:
            x_lo, x_hi = 2, x_wall - 3
        else:
            x_lo, x_hi = x_wall + 4, nx - 3
        if x_hi <= x_lo + 1:
            continue

        # y 位置错开，避免两条带叠在同一行
        span = max(1, (ny - 4) // n_bands)
        y_lo = 2 + k * span + int(rng.integers(0, max(1, span - band_thick)))
        y_lo = int(np.clip(y_lo, 2, ny - band_thick - 2))
        y_hi = y_lo + band_thick

        world.add_box((x_lo, y_lo, 0), (x_hi, y_hi, band_top))
        band_regions.append({
            "x_lo": int(x_lo), "x_hi": int(x_hi),
            "y_lo": int(y_lo), "y_hi": int(y_hi),
            "z_lo": 0, "z_hi": int(band_top),
        })

    # ================================================================
    # 第 2 步：结构建好后再采端点
    # ================================================================
    starts, goals = _sample_endpoints(world, rng, config.n_uavs, clearance, margin)

    def clamp_flyable(p):
        """把端点压到可飞高度内，并吸附到最近的自由格。"""
        c = list(world.world_to_grid(p))
        c[2] = min(c[2], z_fly_max)
        return world.nearest_passable(world.grid_to_world(c), clearance, margin)

    starts = [world.grid_to_world(clamp_flyable(p)) for p in starts]
    goals = [world.grid_to_world(clamp_flyable(p)) for p in goals]

    # ================================================================
    # 第 3 步：可选随机障碍
    # ================================================================
    gap_region = (
        world.origin + np.array([x_wall - 1, gap_y - 1, gap_z - 1]) * res,
        world.origin + np.array([x_wall + 2, gap_y + gap_w + 1, gap_z + gap_h + 1]) * res,
    )

    if config.n_obstacles > 0:
        world.add_random_obstacles(
            config.n_obstacles, rng,
            radius_range=config.obstacle_radius,
            avoid_points=starts + goals,
            corridors=list(zip(starts, goals)),
            corridor_ratio=config.corridor_ratio,
            corridor_offset=config.corridor_offset,
            avoid_regions=[gap_region],
        )

    # ================================================================
    # 第 4 步：验证连通性
    # ================================================================
    scenario = _finish(world, starts, goals, config, "walls", 0)

    scenario.info.update({
        "wall_x": int(x_wall),
        "gap_y": int(gap_y),
        "gap_z": int(gap_z),
        "gap_w": int(gap_w),
        "gap_h": int(gap_h),
        "no_fly_top": int(top_band),
        "z_fly_max": int(z_fly_max),
        "no_fly_bands": band_regions,
    })
    return scenario
