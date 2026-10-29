"""地图插件包。

这个目录下每个 .py 都会被自动加载（见 registry.load_plugins）。
加新地图：新建一个文件，写 @register_map("名字", "中文名") 即可。

参考 walls.py 的写法。最少只需要：

    import numpy as np
    from ..config import ScenarioConfig
    from ..environment import VoxelMap
    from ..registry import register_map
    from ..scenario import Scenario, _finish, _sample_endpoints

    @register_map("mymap", "我的地图")
    def make_mymap(config=None):
        config = config or ScenarioConfig()
        world = VoxelMap(shape=config.planner.shape,
                         resolution=config.planner.resolution)
        world.add_box((10, 10, 0), (15, 15, 8))
        rng = np.random.default_rng(config.seed)
        starts, goals = _sample_endpoints(world, rng, config.n_uavs,
                                           config.planner.hard_clearance, None)
        return _finish(world, starts, goals, config, "mymap", 0)
"""
