"""Safely unpack the repository's own artifact archives."""
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
for archive in sorted((root/'artifacts').glob('*.zip')):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            dest=(root/info.filename).resolve()
            if not dest.is_relative_to(root):raise ValueError(info.filename)
        z.extractall(root)
    print(archive.name)
