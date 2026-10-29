
# UAV 3D Planner 模板工程

一个**可扩展**的三维多无人机路径规划框架。核心设计是**注册表**：
加地图和加算法都只需要新建一个文件，不用改任何原有代码。

自带**浏览器驾驶舱**（3D 轨道相机 + 飞行演示 + 算法演示）和 **matplotlib 单步视图**。

---

## 30 秒上手

`python main.py                     # 打开浏览器驾驶舱（默认入口）
python main.py --step              # A* 单步视图（matplotlib 窗口）
python main.py --info              # 列出所有地图和算法
python main.py --map walls --algorithm rrtstar --plot
python main.py --uavs 1 --obstacles 12 --json

python tests/test_template.py      # 框架自检
python tests/test_step.py          # 单步自检
python tests/test_dashboard.py     # 驾驶舱后端自检`

---

## 目录结构

`模板代码/
├── main.py                      入口（无参数 = 驾驶舱）
├── dashboard.html               驾驶舱页面骨架（无内联 JS）
├── pyproject.toml / requirements.txt
├── examples/demo.py             代码里调用的最小示例
├── tests/
│   ├── test_template.py         框架自检（7 项）
│   ├── test_step.py             单步自检（5 项）
│   └── test_dashboard.py        驾驶舱后端自检（8 项）
└── uav_planner/
    ├── registry.py              ★ 地图注册表（加地图看这里）
    ├── algo_registry.py         ★ 算法注册表（加算法看这里）
    ├── scenario.py              场景公共零件（端点采样、连通性验证）
    ├── environment.py           体素地图、净空栅格、passable 掩码
    ├── geometry.py              解析几何（球/盒/柱 + 连续碰撞）
    ├── config.py                全部参数（唯一真源）
    ├── astar3d.py               A* 实现 + AStarStepper3D 单步状态机
    ├── rrtstar3d.py             RRT* 实现
    ├── smoothing.py             碰撞感知平滑
    ├── guidance.py              动态引导点
    ├── dynamics.py              限速限加速模型
    ├── fleet.py                 组合层：规划 -> 平滑 -> rollout
    ├── metrics.py               规划层/执行层双份指标
    ├── dashboard.py             驾驶舱 HTTP 服务
    ├── visualization.py         静态 3D 图 + matplotlib 单步视图
    ├── cli.py                   命令行
    ├── web/viewer.js            Canvas 2D 渲染引擎（轨道相机 + 画家算法）
    ├── maps/walls.py            ★ 地图插件：缺口墙
    └── algos/{astar,rrtstar}.py ★ 算法插件`

---

## 浏览器驾驶舱

`python main.py                 # 自动开浏览器，默认 http://127.0.0.1:8899
python main.py --port 9000     # 换端口
python main.py --no-browser    # 不自动开

**端口说明**：默认 8899，避开 8765（很多实验台/工具默认占用）。
如果 8899 也被占用，服务会自动往后找 8899~8918 之间第一个空闲端口，
并在终端打印实际地址，不会直接崩掉。`

### 界面结构

| 区域 | 内容 |
|---|---|
| 左栏 | 地图、算法、计算设置（尺寸/启发式/对角/平滑）、种子、机数、障碍数、硬净空 |
| 中栏 | 3D 画布 + 两种模式切换 + 视角预设 + 轨迹回放条 |
| 右栏 | 核心指标、规划层/执行层对照、路径质量条、实验说明 |

### 两种模式

**飞行演示**：显示规划路径（青色）+ 执行轨迹（绿色），可拖进度条回放。

**算法演示**：A* 搜索过程逐帧播放，橙色=open 待探索，蓝色=closed 已探索，
白色=当前节点，右侧实时显示 **g / h / f = g+h**、扩展数、frontier 大小。

### 鼠标操作

| 操作 | 作用 |
|---|---|
| 左键拖拽 | 旋转（yaw + pitch，**竖直拖拽真的会俯仰**） |
| 右键 / Shift+左键拖拽 | 平移 |
| 滚轮 | 缩放 |
| 双击 / R | 重置视角 |
| 方向键 | 旋转 |
| +/- | 缩放 |

视角预设按钮：等轴测 / 俯视 / 正视 / 侧视 / 重置。

### 3D 渲染原理（viewer.js）

**轨道相机**：球坐标定位，透视投影。

`eye = target + distance * [cos(pitch)sin(yaw), -cos(pitch)cos(yaw), sin(pitch)]
f = normalize(target - eye)          // 前
r = normalize(cross(f, [0,0,1]))     // 右（世界 up = +Z）
u = cross(r, f)                      // 上

// 投影
zv = dot(d, f);   if (zv < near) return null;   // 背面裁剪
screen = [W/2 + dot(d,r)*focal/zv, H/2 - dot(d,u)*focal/zv, zv]`

**pitch 必须 clamp 在 [0.02, 1.50]**：pitch = ±π/2 时视线与 up 轴平行，
cross(f, up) 退化成零向量，normalize 后变 NaN，整幅画面会消失。

**画家算法**：所有图元（立方体面、球、线段、标记）塞进同一个 prims 数组，
按深度从远到近绘制。这样路径才会被建筑正确遮挡，而不是永远浮在最上层。

---

## ★ 加自己的地图（2 步，不改任何原文件）

### 第 1 步：新建 uav_planner/maps/mymap.py

`import numpy as np

from ..config import ScenarioConfig
from ..environment import VoxelMap
from ..registry import register_map
from ..scenario import Scenario, _finish, _sample_endpoints


`register_map("mymap", "我的地图")
def make_mymap(config: ScenarioConfig | None = None) -> Scenario:
    config = config or ScenarioConfig()
    p = config.planner

    world = VoxelMap(shape=p.shape, resolution=p.resolution)

    # 加障碍：只用 add_box / add_sphere / add_cylinder
    world.add_box((10, 10, 0), (15, 15, 8))         # 楼
    world.add_sphere((20, 20, 4), radius=2.0)       # 球
    world.add_cylinder((25, 10), 0, 6, radius=1.5)  # 柱

    # 采起终点（也可以自己写死坐标）
    rng = np.random.default_rng(config.seed)
    starts, goals = _sample_endpoints(world, rng, config.n_uavs,
                                       p.hard_clearance, None)

    # 验证连通性 + 自动修复（必须调用）
    return _finish(world, starts, goals, config, "mymap", 0)`

### 第 2 步：直接用

`python main.py --map mymap`

**驾驶舱的下拉框会自动出现你的地图** —— 前端启动时 fetch('/api/meta')
动态生成选项，不用改 dashboard.html。

---

## ★ 加自己的算法（2 步，同样不改原文件）

### 第 1 步：新建 uav_planner/algos/myalgo.py

`import numpy as np

from ..algo_registry import register_algo
from ..astar3d import CostWeights, PathResult
from ..environment import Cell, VoxelMap


`register_algo("myalgo", "我的算法")
def plan_myalgo(world: VoxelMap, start: Cell, goal: Cell,
                weights: CostWeights, **kwargs) -> PathResult:
    # 在这里写你的搜索逻辑
    points = np.stack([world.grid_to_world(start), world.grid_to_world(goal)])
    length = float(np.linalg.norm(points[-1] - points[0]))

    return PathResult(
        points=points,
        cost=length,
        expanded=0,
        success=True,
        planner="myalgo",
        metadata={
            "length": length,
            "min_clearance": world.path_clearance(points),
            "waypoints": float(len(points)),
            "hard_clearance": weights.hard_clearance,
        },
    )`

### 第 2 步：直接用

`python main.py --algorithm myalgo`

驾驶舱的算法下拉框同样会自动出现。

---

## 算法契约

`plan(world, start, goal, weights, **kwargs) -> PathResult`

| 参数 | 类型 | 说明 |
|---|---|---|
| world | VoxelMap | 体素地图 |
| start / goal | (i,j,k) int 元组 | **格子索引**，不是世界坐标 |
| weights | CostWeights | 含 hard_clearance 硬约束 |
| kwargs | — | 会收到 diagonal / max_expansions / heuristic / safety_margin / other_paths / seed |

返回 PathResult：

| 字段 | 说明 |
|---|---|
| points | (N,3) 世界坐标 |
| cost | 总代价 |
| expanded | 扩展节点数 |
| planner | 名字，会显示在指标里 |
| metadata | 至少含 length / min_clearance / waypoints / hard_clearance |

**多机是免费的**：plan_fleet() 会依次规划每架，把前面的路径作为
other_paths 传给你的算法。你只要接收并处理这个参数就行。

---

## 地图契约

返回 Scenario(world, starts, goals, info)。

### 三条硬性要求

| 要求 | 不满足会怎样 |
|---|---|
| 端点必须在 passable 掩码内 | ValueError: start violates the hard clearance constraint |
| 起终点必须连通 | RuntimeError: no feasible 3D path found |
| 障碍必须用 add_* 加 | **A* 完全看不见**，直接穿墙 |

### 最大的坑

`world.occupied[5,5,0:4] = True   # 错误！`

occupied 只用于**显示和距离场**，A* 判断能不能走用的是 obstacles ->
clearance_grid() -> passable()。只改 occupied 会导致净空报告 inf、A* 穿墙。

**正确做法**：world.add_box(...) / add_sphere(...) / add_cylinder(...)。

---

## 单步视图（matplotlib 版）

`python main.py --step`

| 按键 | 作用 |
|---|---|
| 空格 / → | 扩展一个节点 |
| a | 一路跑到终点 |
| r | 重置 |
| h | 切换启发函数（A* <-> Dijkstra） |
| s | 切换每次按键扩展数（1/2/4/8） |
| q | 关闭窗口 |

实时显示当前节点的 g / h / f / 净空，以及 open / closed 规模。

**为什么它可信**：AStarStepper3D 和 AStar3D 共享同一个 _LookupGrid，
所以屏幕上显示的数值就是真实规划器内部的值。
test_stepper_matches_planner 强制校验两者的扩展数和路径完全一致。

> 注：驾驶舱里的「算法演示」模式是同一个单步状态机的 Web 版，
> 用 python main.py 打开就能看到。

---

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| --map | walls | 地图名 |
| --algorithm | astar | 算法名 |
| --seed | 7 | 随机种子（同 seed 结果位一致） |
| --uavs | 3 | 无人机数量 |
| --obstacles | 0 | 额外随机障碍数 |
| --shape NX NY NZ | 32 32 14 | 网格尺寸 |
| --resolution | 1.0 | 每格米数 |
| --hard | 0.35 | 硬净空约束（米） |
| --clearance | 2.0 | 软净空半径（米） |
| --min-separation | 2.5 | 多机最小间距 |
| --heuristic | euclidean | euclidean=A*，zero=Dijkstra |
| --no-diagonal | — | 退化为 6 邻域 |
| --no-smooth | — | 跳过平滑 |
| --step | — | matplotlib 单步视图 |
| --dashboard | — | 浏览器驾驶舱（无参数时默认） |
| --port | 8899 | 驾驶舱端口（被占用时自动 +1 重试） |
| --no-browser | — | 不自动开浏览器 |
| --plot | — | 画 3D 图 |
| --json | — | 输出 JSON |
| --info | — | 列出所有地图和算法 |

---

## 自带的地图：缺口墙 + 禁飞区

这张图由**三个结构**叠加而成，每个都有明确的设计意图：

### 1. 缺口墙（竖直，横跨 y-z 截面）

一面墙横跨整个 y-z 截面，只留一个矩形洞，洞的位置由 seed 决定。

**为什么需要它**：26 邻域搜索**无法在 xy 平面内绕过**这面墙，
所以路径必须使用 z 轴 —— 它强制结果是"真三维"的，
而不是在一层薄薄的平面上打转。

### 2. 顶部禁飞层（横向，铺满整个 x-y 平面）

在可飞高度之上封 3 层，**铺满整个 x-y 平面**。

**为什么需要它**：如果墙只到某个高度，飞行器完全可以爬升越过墙顶走捷径。
加了禁飞层后，"飞过去"这条路被彻底封死，只能老老实实钻洞。
这是对"上方空间要严谨"的直接回应。

### 3. 横向禁飞区（沿 x 贯穿的条带，但只到中等高度）

两条条带，分别位于墙的左右两侧，y 位置错开。

**为什么只到 70% 高度而不是全高**：如果做成全高，整张地图会被切成
互不连通的区域 —— 那就不是"规划问题"而是"无解问题"了。
只到中等高度时，路径必须做**垂直 + 水平**的组合绕行：
要么从条带上方越过，要么从旁边绕过。这才真正检验规划器的三维能力。

### 视觉表现

禁飞区在驾驶舱里渲染成**半透明幽灵盒**：

| 类型 | 颜色 | 面 alpha | 棱 alpha |
|---|---|---|---|
| 顶部禁飞层 | 红 #ff6b7a | 0.045 | 0.30 |
| 横向禁飞带 | 橙 #ffb454 | 0.06 | 0.42 |

面几乎透明，只保留棱线勾出体积边界，不会挡住里面的路径。

### 实现要点

**必须先建结构，再采端点**。如果先采端点后加禁飞层，
端点可能正好落在禁飞结构里，随后 `_finish()` 会判定不连通并疯狂删障碍，
把设计好的结构全删光（我踩过这个坑，`repairs` 会变成 3）。

墙的构造用**补集法**：整面墙减去一个洞 = 4 块互不重叠的矩形，
每块都检查非空后才添加（Box 要求 max 严格大于 min）。

---

## 自带的算法

| 名字 | 特点 |
|---|---|
| astar | 26 邻域多目标 A*，硬净空约束 + 多机分离代价，10 µs/次扩展 |
| rrtstar | 连续空间 RRT*，渐近最优，不处理多机分离 |

---

## 两个关键设计

**1. 为什么 A* 内层可以只查表？**

passable(clearance, margin) 要求每个格子中心到障碍 >= clearance + margin。
解析距离是 1-Lipschitz 的，所以取 margin = 0.5·√3·resolution 时，
**格子内任意一点、任意 26 邻域边上任意一点**都 >= clearance。

硬约束由掩码本身保证，内层不需要逐边采样。几何计算只在建图和最终校验各做一次。

**2. 为什么执行层守得住硬净空？**

两个机制叠加：

- GuidanceTracker.effective_lookahead() 在净空小时**收缩前视距离**（切角量级≈前视距离）
- KinematicModel._safe_fraction() 每步**二分**找最大安全比例

所以报告的 min_clearance 不会低于约束值。

---

## 已验证

`tests/test_template.py   ->  7 passed, 0 failed
tests/test_step.py       ->  5 passed, 0 failed
tests/test_dashboard.py  ->  8 passed, 0 failed
                            （合计 20 项）

HTTP 接口实测：
  GET /              200, 9353 bytes
  GET /viewer.js     200, 25474 bytes
  GET /api/meta      200, maps=[walls], algorithms=[astar, rrtstar]
  GET /api/plan      200, paths=1, clearance=0.811
  GET /api/step      200, frames=366, expanded=365

默认运行（walls + astar + 3 架）：
  规划层净空 0.47 m >= 0.35 m  OK
  执行层净空 0.47 m >= 0.35 m  OK
  mean_path_ratio 1.16（有绕行）

单步视图（walls + seed 7，单机）：
  A* 扩展 365 次，Dijkstra 扩展 9618 次
  单步扩展数与真实 A* 完全一致，路径完全一致`

---

## 后续可以加什么

| 方向 | 做法 |
|---|---|
| 更多地图 | maps/ 下新建文件 |
| 更多算法（D* Lite / Hybrid A* / CBS） | algos/ 下新建文件 |
| 算法对比模式 | 复用 /api/plan 循环调多个算法，前端叠路径 |
| 动态重规划 | 在 algos/ 加一个带状态缓存的插件 |
| 强化学习 | 保留 environment.py + dynamics.py，自己写 env |
| 视频导出 | 前端 canvas.captureStream + MediaRecorder |

模板**不含**：算法对比、视频录制。
这些是展示功能，需要时基于现有 /api/plan 和 metrics 接口加即可。
