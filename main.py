"""入口。

不带参数运行 = 打开浏览器驾驶舱（推荐）。
带参数透传给 uav_planner.cli。

常用：
    python main.py                                  # 驾驶舱
    python main.py --no-browser                     # 驾驶舱（不开浏览器）
    python main.py --port 9000                      # 驾驶舱（指定端口）
    python main.py --info                           # 列出地图和算法
    python main.py --step                           # A* 单步视图（matplotlib）
    python main.py --map walls --algorithm rrtstar --plot
    python main.py --uavs 1 --obstacles 12 --json
"""

import sys

from uav_planner.cli import main

# 这些参数本身就意味着"要开驾驶舱"，即使没显式写 --dashboard
DASHBOARD_HINTS = ("--no-browser", "--port")


if __name__ == "__main__":
    argv = sys.argv[1:]

    # 无参数 -> 驾驶舱
    if not argv:
        argv = ["--dashboard"]

    # 只给了驾驶舱相关参数（如 --port 9000）-> 补上 --dashboard
    elif ("--dashboard" not in argv
          and any(a.split("=")[0] in DASHBOARD_HINTS for a in argv)
          and not any(a.startswith("--") and a.split("=")[0] not in DASHBOARD_HINTS
                      for a in argv)):
        argv = ["--dashboard"] + argv

    raise SystemExit(main(argv))
