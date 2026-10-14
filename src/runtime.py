"""Local runtime bridge; normal Python installations need no path overrides."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
for path in [Path(r'D:\PycharmProjects\KAGGLE_airline\.venv\Lib\site-packages'),Path(r'D:\kaggle\airline\.venv_ml\Lib\site-packages')]:
    if path.exists():sys.path.append(str(path))
