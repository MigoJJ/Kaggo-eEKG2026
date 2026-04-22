import json
import zipfile
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="ascii") as handle:
        return json.load(handle)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_archive_if_needed(archive_path, extract_root):
    archive_path = Path(archive_path)
    extract_root = Path(extract_root)
    ensure_dir(extract_root)

    with zipfile.ZipFile(archive_path) as zf:
        names = [name for name in zf.namelist() if name and not name.endswith("/")]
        top_levels = sorted({name.split("/")[0] for name in names})
        if not top_levels:
            raise RuntimeError(f"Archive has no files: {archive_path}")
        top_level = top_levels[0]
        target_dir = extract_root / top_level
        sentinel = target_dir / ".extracted"
        if not sentinel.exists():
            zf.extractall(extract_root)
            sentinel.write_text("ok\n", encoding="ascii")
    return target_dir
