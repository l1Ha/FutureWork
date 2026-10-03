import sys
import os

# 确保捆绑包解压路径或当前路径被包含
if hasattr(sys, '_MEIPASS'):
    sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from futurework.cli import main

if __name__ == "__main__":
    sys.exit(main())
