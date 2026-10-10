import os
from pathlib import Path


def get_working_directory(path: str = "books") -> str:
    return os.path.join(str(Path.cwd().resolve()), path)


def get_extracted_pages_dir(path: str = "output/") -> Path:
    return Path(get_working_directory(path=path))


def read_file_contents(filepath: str | Path) -> str:
    return Path(filepath).read_text(encoding="utf-8")
