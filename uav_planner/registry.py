"""场景注册表 —— 插件式地图系统。

这是整个模板的扩展核心。你以后加地图，只需要：

    1. 在 uav_planner/maps/ 下新建一个 .py 文件
    2. 写一个 @register_map("名字") 装饰的函数
    3. 什么都不用改

本文件不做任何几何计算，只负责"按名字找到 builder"。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .config import ScenarioConfig
from .environment import VoxelMap
from .scenario import Scenario, _finish, _sample_endpoints

# 场景构造函数签名：接收配置，返回 Scenario
MapBuilder = Callable[[ScenarioConfig], Scenario]

# 全局注册表：名字 -> 构造函数
_MAPS: Dict[str, MapBuilder] = {}

# 中文显示名（可选，给前端下拉框用）
_LABELS: Dict[str, str] = {}


def register_map(name: str, label: Optional[str] = None):
    """装饰器：把一个场景函数注册到全局表。

    用法::

        @register_map("walls", "缺口墙")
        def make_walls(config):
            ...

    重复注册同名会抛 ValueError，避免静默覆盖。
    """
    def decorator(fn: MapBuilder) -> MapBuilder:
        if name in _MAPS:
            raise ValueError(f"map '{name}' is already registered")
        _MAPS[name] = fn
        _LABELS[name] = label or name
        return fn
    return decorator


def get_map(name: str) -> MapBuilder:
    """按名字取构造函数，找不到抛 KeyError（附可用列表）。"""
    try:
        return _MAPS[name]
    except KeyError:
        raise KeyError(f"unknown map {name!r}; available: {sorted(_MAPS)}") from None


def has_map(name: str) -> bool:
    return name in _MAPS


def list_maps() -> tuple:
    """所有已注册地图名（按注册顺序）。"""
    return tuple(_MAPS)


def map_labels() -> Dict[str, str]:
    """名字 -> 中文显示名。"""
    return dict(_LABELS)


def build(name: str, config: Optional[ScenarioConfig] = None) -> Scenario:
    """按名字构造场景。"""
    return get_map(name)(config or ScenarioConfig())


def load_plugins() -> None:
    """自动导入 maps/ 目录下所有模块，触发装饰器注册。

    在包初始化时调用一次即可。找不到目录也不报错。
    """
    import importlib
    import pkgutil

    package = importlib.import_module(__package__ + ".maps")
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{package.__name__}.{info.name}")


__all__ = [
    "MapBuilder", "register_map", "get_map", "has_map", "list_maps",
    "map_labels", "build", "load_plugins", "Scenario", "VoxelMap",
    "_finish", "_sample_endpoints",
]
