from contextlib import contextmanager
from pathlib import Path


@contextmanager
def numbered_output(root: Path, prefix: str, folders: tuple[str, ...] = ("",)):
    """Reserve a readable number across processes without replacing existing files."""
    root.mkdir(parents=True, exist_ok=True)
    maximum = 0
    for folder in folders:
        directory = root / folder
        if directory.is_dir():
            for path in directory.iterdir():
                suffix = path.stem.removeprefix(prefix)
                if path.stem.startswith(prefix) and suffix.isdigit():
                    maximum = max(maximum, int(suffix))
    number = maximum + 1
    while True:
        name = f"{prefix}{number:06d}"
        lock = root / f".{name}.lock"
        try:
            with lock.open("x"):
                pass
        except FileExistsError:
            number += 1
            continue
        # Recheck after reserving: another writer may have just completed.
        if any(any((root / folder).glob(f"{name}.*")) or (root / folder / name).exists()
               for folder in folders):
            lock.unlink()
            number += 1
            continue
        break
    try:
        yield name
    finally:
        lock.unlink(missing_ok=True)
