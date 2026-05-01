import pkgutil
import importlib
from pathlib import Path


# 获取当前包的路径
_package_path = [str(Path(__file__).parent)]
_package_name = __name__

# 递归遍历当前包下的所有子模块
for loader, module_name, is_pkg in pkgutil.walk_packages(
    _package_path, _package_name + "."
):
    importlib.import_module(module_name)
