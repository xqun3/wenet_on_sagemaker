# __init__.py
import os
import sys

# 将项目根目录添加到 Python 路径
# project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(project_root)
# print(sys.path)
from wenet.cli.model import load_model, load_model_pt  # noqa
