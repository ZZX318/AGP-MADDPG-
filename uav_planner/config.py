"""配置对象 —— 全部参数集中在这里。

参数严格对齐论文
----------------
H. Wang, X. Sun, Y. Liu, C. Mu,
"AGP-MADDPG: A* Guidance Point based Multi-Agent Deep Deterministic Policy
Gradient for 3D cooperative multi-UAV trajectory planning",
Expert Systems With Applications 333 (2027) 134264.

对应关系
--------
Table 1（参考路径规划 + 奖励整形参数）
    lambda_obs = 0.35     障碍代价权重
    lambda_sep = 0.05     机间分离代价权重
    d_safe     = 0.5      障碍安全阈值
    sigma_sep  = 0.3      分离核带宽
    alpha      = 2.0      引导奖励增益
    beta       = -0.8     障碍接近惩罚系数
    eta        = 1.5      碰撞惩罚系数
    tau        = 0.5      机间间距惩罚系数
    R_g        = 1.5      引导感知半径
    R_o        = 2.0      障碍安全缓冲
    d_0        = 1.0      最小机间距离

Sec. 3.2（改进 A*）
    分辨率 delta = 0.15 m
    邻域 26
    f_ours(n) = g(n) + h(n) + lambda_obs * c_obs(n) + lambda_sep * c_sep(n)
    c_obs(n) = [max(0, 1 - d_obs(n)/d_safe)]^2
    c_sep(n) = sum_p exp(-||x(n) - p||^2 / (2 * sigma_sep^2))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CostWeights:
    """改进 A* 的代价系数（论文式 2/3/4）。

    length      -> g(n) 里的步长系数，论文取 1
    obstacle    -> lambda_obs，障碍代价权重
    separation  -> lambda_sep，机间分离代价权重
    clearance   -> d_safe，障碍安全阈值
    min_separation -> d_0，最小机间距离
    hard_clearance -> 硬约束下限（论文用软惩罚，这里额外保留）
    """

    length: float = 1.0
    obstacle: float = 0.35        # lambda_obs
    separation: float = 0.05      # lambda_sep
    clearance: float = 0.5        # d_safe
    minimum_separation: float = 1.0   # d_0
    hard_clearance: float = 0.35
    sigma_sep: float = 0.3            # sigma_sep（式 4 的核带宽）

    def replace(self, **changes) -> "CostWeights":
        from dataclasses import replace
        return replace(self, **changes)


@dataclass
class PlannerConfig:
    """改进 A* 的参数。

    resolution 论文取 0.15 m（Sec. 3.2）。
    safety_margin 为 None 时取半个体素对角线，保证 26 邻域安全。
    """

    resolution: float = 0.15                  # delta
    shape: Tuple[int, int, int] = (64, 64, 24)

    length_weight: float = 1.0
    obstacle_weight: float = 0.35             # lambda_obs
    separation_weight: float = 0.05           # lambda_sep

    # d_safe（式 3 的障碍安全阈值）。
    # 必须 > hard_clearance + safety_margin，否则 c_obs 永远为 0：
    # passable 掩码已经保证每个可通行格子的净空 >= hard + margin，
    # 若 d_safe 不比这个大，障碍代价就永远不会触发，消融实验也看不出差别。
    # 论文里 d_safe = 0.5、没有硬约束，所以不受影响；
    # 这里保留硬约束，就把 d_safe 放大到能覆盖"安全缓冲带"的量级。
    clearance: float = 2.0                    # d_safe

    hard_clearance: float = 0.35              # 硬下限（论文用软惩罚，这里额外保留）
    minimum_separation: float = 1.0           # d_0

    sigma_sep: float = 0.3                    # sigma_sep（分离核带宽）
    max_expansions: int = 500_000
    diagonal_cost: bool = True
    heuristic: str = "euclidean"
    safety_margin: Optional[float] = None

    def to_weights(self) -> CostWeights:
        return CostWeights(
            length=self.length_weight,
            obstacle=self.obstacle_weight,
            separation=self.separation_weight,
            clearance=self.clearance,
            minimum_separation=self.minimum_separation,
            hard_clearance=self.hard_clearance,
            sigma_sep=self.sigma_sep,
        )


@dataclass
class DynamicsConfig:
    """点质量运动模型（论文未给具体参数，取合理值）。"""

    dt: float = 0.1
    max_speed: float = 2.0
    max_acceleration: float = 2.0
    goal_tolerance: float = 0.3
    lookahead: float = 1.0
    hard_clearance: float = 0.35


@dataclass
class RewardConfig:
    """奖励项，严格对应论文式 (7)-(11)。

    r_i = r_arrive + r_guide + r_uav-uav + r_obstacle + r_collision
    """

    # r_arrive：到达奖励（论文未给数值，取正数）
    arrival: float = 20.0

    # r_guide = alpha * max(0, 1 - (d(P_guide, P_UAV)/R_g)^2)     式 (7)
    guidance_gain: float = 2.0        # alpha
    guidance_radius: float = 1.5      # R_g

    # r_obstacle = beta * (1/(d_o - R_obstacle) - 1/R_o)          式 (8)
    obstacle_penalty: float = -0.8    # beta
    obstacle_buffer: float = 2.0      # R_o

    # r_collision = eta * ((d_o - R_obstacle)/R_obstacle - 1)     式 (9)
    collision_penalty: float = 1.5    # eta

    # r_uav-uav = tau * (d_u - d_0)                              式 (10)
    separation_penalty: float = 0.5   # tau
    min_uav_distance: float = 1.0     # d_0


@dataclass
class ScenarioConfig:
    """可复现的场景参数。"""

    seed: int = 7
    n_uavs: int = 1                   # 前期复现：单机
    n_obstacles: int = 12
    n_walls: Optional[int] = None
    obstacle_radius: Tuple[float, float] = (1.5, 3.0)
    corridor_ratio: float = 0.55
    corridor_offset: float = 3.0
    max_repairs: int = 64
    planner: PlannerConfig = field(default_factory=PlannerConfig)
