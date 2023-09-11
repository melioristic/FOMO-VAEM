from pathlib import Path

def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

