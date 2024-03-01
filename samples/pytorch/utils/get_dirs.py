import os
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = os.path.abspath(os.path.join(root_dir, "data"))
media_dir = os.path.abspath(os.path.join(data_dir, "media"))
models_dir = os.path.abspath(os.path.join(data_dir, "models"))
