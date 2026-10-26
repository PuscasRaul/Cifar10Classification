#!/usr/bin/env python3
import os
import glob as gl
from src.core.CifarDataset import CifarDataset

database_root_dir = os.path.abspath(os.environ["DATABASE_DIR"])
databse_files = gl.glob(database_root_dir + "/" + "data_batch_[1-9]")
loaded_db = [CifarDataset(file) for file in databse_files]
