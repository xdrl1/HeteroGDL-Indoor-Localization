import os
import shutil

from pathlib import Path


def reload_directory(dir):
    dirpath = Path(dir)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.mkdir(dir)
