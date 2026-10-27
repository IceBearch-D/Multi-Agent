"""根目录一键入口（等价于 run.sh）。

让用户能够用 `python run.py` 直接启动命令行工具，而无需关心 src 目录的路径问题。

原理：
  1. 把项目根目录下的 src/ 插入 sys.path，使 `from main import main` 可用；
  2. 调用 src/main.py 中的 main() 函数并传递退出码。
"""
import sys  # sys.path 操作与退出码
from pathlib import Path  # 解析路径

# 把 src 目录加入 Python 模块搜索路径，这样下面的 `from main import main` 才能找到
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from main import main  # 导入命令行主入口

if __name__ == "__main__":  # 作为脚本运行时
    sys.exit(main())       # 以 main 的返回值作为进程退出码