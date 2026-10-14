"""Restore raw competition/original CSVs from the repository gzip files."""
from pathlib import Path
import gzip, shutil
root=Path(__file__).resolve().parents[1]
for source in (root/'data').glob('*.csv.gz'):
    dest=source.with_suffix('')
    if not dest.exists():
        with gzip.open(source,'rb') as src,dest.open('wb') as out:shutil.copyfileobj(src,out)
        print(dest.name)
