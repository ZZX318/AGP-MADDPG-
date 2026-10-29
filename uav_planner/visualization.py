
"""可视化：静态 3D 图 + A* 单步交互视图。

单步视图的设计要点（沿用原项目，很有教学价值）：
* 它和真实规划器共享同一个 _LookupGrid，所以屏幕上显示的 g/h/f
  就是规划器内部真实的数值，不是"演示专用"的假数据；
* 支持现场切换启发函数（A* <-> Dijkstra），直观看到扩展数的差别。
"""

from __future__ import annotations

import numpy as np

# 调色板
COLOR_CLOSED = "#9aa5b1"
COLOR_OPEN = "#f0a020"
COLOR_CURRENT = "#d62728"
COLOR_PATH = "#2ca02c"
COLOR_OBSTACLE = "#e07070"


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "画图需要 matplotlib，安装：pip install -e '.[plot]'") from exc
    return plt


def _cjk(size: float = 9.0, weight: str = "normal"):
    """挑一个能画中文的字体。"""
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                   "Arial Unicode MS", "Hiragino Sans GB", "STHeiti",
                   "WenQuanYi Micro Hei"):
        if family in installed:
            return FontProperties(family=[family, "DejaVu Sans"],
                                  size=size, weight=weight)
    return FontProperties(size=size, weight=weight)


def _apply_cjk_rc(plt):
    families = list(_cjk().get_family())
    plt.rcParams["font.sans-serif"] = families
    plt.rcParams["axes.unicode_minus"] = False


def _set3d(collection, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    collection._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
    return collection


def _draw_obstacles(ax, world, alpha: float = 0.16):
    """每个占据体素画一个半透明方块。"""
    cells = np.argwhere(world.occupied)
    if len(cells) == 0:
        return None
    points = world.origin + cells * world.resolution
    return ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      s=max(4.0, 220.0 * world.resolution ** 2 / 8.0),
                      c="tab:red", alpha=alpha, marker="s",
                      linewidths=0, depthshade=False)


def _setup_axes(ax, world, title: str):
    ax.set_xlim(world.origin[0], world.maximum[0])
    ax.set_ylim(world.origin[1], world.maximum[1])
    ax.set_zlim(world.origin[2], world.maximum[2])
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    ax.set_title(title)


def plot_scenario(world, paths=None, starts=None, goals=None, show: bool = True):
    """画静态 3D 图：障碍 + 路径 + 起终点。"""
    plt = _plt()
    _apply_cjk_rc(plt)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    _draw_obstacles(ax, world)

    for i, path in enumerate(paths or []):
        path = np.asarray(path, dtype=float)
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                linewidth=2, label=f"无人机 {i}")

    if starts is not None and len(starts):
        s = np.asarray(starts, dtype=float)
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], marker="o", c="green",
                   s=45, label="起点")
    if goals is not None and len(goals):
        g = np.asarray(goals, dtype=float)
        ax.scatter(g[:, 0], g[:, 1], g[:, 2], marker="*", c="black",
                   s=70, label="目标")

    _setup_axes(ax, world, "3D 多无人机路径")
    if paths or starts is not None:
        ax.legend(loc="upper left")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def interactive_astar(world, start, goal, weights=None, heuristic: str = "euclidean",
                      diagonal: bool = True, safety_margin=None,
                      steps_per_key: int = 1, max_steps_per_frame: int = 2_000_000):
    """A* 单步交互视图，实时显示当前节点的 g / h / f。

    按键
    ----
    space / ->  扩展一个节点（或 steps_per_key 个）
    a           一路跑到终点
    r           重置
    h           切换启发函数（A* <-> Dijkstra，h=0 就是 Dijkstra）
    s           切换每次按键扩展的节点数（1/2/4/8）
    q           关闭窗口

    返回驱动它的 AStarStepper3D 实例，方便测试。
    """
    plt = _plt()
    _apply_cjk_rc(plt)
    from .astar3d import AStarStepper3D

    stepper = AStarStepper3D(world, start, goal, weights, heuristic=heuristic,
                             diagonal=diagonal, safety_margin=safety_margin)

    fig = plt.figure(figsize=(12.6, 8.6))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.22)
    ax.set_box_aspect((4, 4, 3), zoom=1.18)
    _draw_obstacles(ax, world)

    closed = _set3d(ax.scatter([], [], [], s=7, c=COLOR_CLOSED, depthshade=False), [])
    opened = _set3d(ax.scatter([], [], [], s=9, c=COLOR_OPEN, depthshade=False), [])
    current = _set3d(ax.scatter([], [], [], s=90, c=COLOR_CURRENT,
                                marker="o", depthshade=False), [])
    path_line, = ax.plot([], [], [], linewidth=2.4, c=COLOR_PATH)

    start_pt = np.asarray(world.grid_to_world(start), float)
    goal_pt = np.asarray(world.grid_to_world(goal), float)
    ax.scatter(*start_pt, marker="o", c="green", s=70, edgecolors="black", linewidths=0.6)
    ax.scatter(*goal_pt, marker="*", c="black", s=130)
    _setup_axes(ax, world, "A* 单步")

    # 图例：四种节点状态 + 障碍 + 起终点
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=COLOR_PATH, lw=2.4, label="绿线 找到的路径"),
        Line2D([], [], marker="o", ls="", color=COLOR_OPEN, label="橙点 open 待考察"),
        Line2D([], [], marker="o", ls="", color=COLOR_CLOSED, label="灰点 closed 已扩展"),
        Line2D([], [], marker="o", ls="", color=COLOR_CURRENT, ms=9, label="红点 当前节点"),
        Line2D([], [], marker="s", ls="", color=COLOR_OBSTACLE, ms=8, label="粉块 障碍体素"),
        Line2D([], [], marker="o", ls="", color="green", label="绿圆 起点"),
        Line2D([], [], marker="*", ls="", color="black", ms=11, label="黑星 目标"),
    ], loc="upper right", prop=_cjk(9), framealpha=0.92,
        title="图例", title_fontproperties=_cjk(9, "bold"))

    info = fig.text(0.74, 0.02, "", va="bottom", ha="left",
                    fontproperties=_cjk(9.5),
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                              edgecolor="#c8c8c8", alpha=0.94))

    fig.text(0.03, 0.02,
             "按键   空格/-> 下一步    a 跑到终点    r 重置    h 切换启发函数（A* <-> Dijkstra）\n"
             "       s 每次按键扩展数（1/2/4/8）    q 关闭窗口\n"
             "怎么看  每步从 open 里取 f 最小的节点 -> 标 closed -> 扩展邻居 -> 放回 open；f = g + h",
             va="bottom", ha="left", fontproperties=_cjk(9.5), linespacing=1.7,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7",
                       edgecolor="#c8c8c8", alpha=0.98))

    state = {"per_key": steps_per_key}

    def refresh():
        def cells(indices):
            if not indices:
                return np.zeros((0, 3))
            return np.stack([world.grid_to_world(stepper.table.cell_of(i)) for i in indices])

        _set3d(closed, cells(list(stepper.closed)))
        _set3d(opened, cells([i for i in stepper.open_set if i not in stepper.closed]))
        if stepper.current is not None:
            _set3d(current, cells([stepper.current]))
        if len(stepper.path):
            pts = np.asarray(stepper.path, float)
            path_line.set_data(pts[:, 0], pts[:, 1])
            path_line.set_3d_properties(pts[:, 2])

        if stepper.current is not None:
            info_node = stepper.node_info(stepper.current)
            x, y, z = stepper.table.cell_of(stepper.current)
            body = (f"节点 ({x},{y},{z})\n"
                    f"g = {info_node['g']:.2f}   已走代价\n"
                    f"h = {info_node['h']:.2f}   估计剩余\n"
                    f"f = {info_node['f']:.2f}   f = g + h\n"
                    f"净空 = {info_node['clearance']:.2f} m\n"
                    f"open {len(stepper.open_set)}    closed {len(stepper.closed)}")
        else:
            body = "未开始\n按 [空格] 扩展第一个节点"
        info.set_text(body)

        w = stepper.weights
        ax.set_title(f"A* 单步 - 已扩展 {stepper.expanded} 个节点"
                     + ("  ✓ 已到达" if stepper.reached else "")
                     + ("" if stepper.frontier or stepper.done else "  (open 已空)")
                     + "\n"
                     + f"h={stepper.heuristic_mode}   邻域={'26' if stepper.diagonal else '6'}"
                     + f"   w_obs={w.obstacle:.1f}   w_clear={w.clearance:.1f}"
                     + f"   hard={w.hard_clearance:.2f}   min_sep={w.minimum_separation:.1f}"
                     + f"   每次按键={state['per_key']}",
                     fontproperties=_cjk(11), linespacing=1.6)
        fig.canvas.draw_idle()

    def advance(count: int):
        for _ in range(max(1, count)):
            if stepper.done:
                break
            stepper.step()
        refresh()

    def on_key(event):
        key = event.key
        if key in (" ", "right", "n"):
            advance(state["per_key"])
        elif key == "a":
            for _ in range(max_steps_per_frame):
                if stepper.done:
                    break
                stepper.step()
            refresh()
        elif key == "r":
            stepper.reset()
            path_line.set_data([], [])
            path_line.set_3d_properties([])
            refresh()
        elif key == "h":
            stepper.set_heuristic("zero" if stepper.heuristic_mode == "euclidean"
                                  else "euclidean")
            refresh()
        elif key == "s":
            state["per_key"] = 1 if state["per_key"] >= 8 else state["per_key"] * 2
            refresh()
        elif key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()
    return stepper


__all__ = ["plot_scenario", "interactive_astar"]
