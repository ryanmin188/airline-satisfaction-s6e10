"""Build lossless artifact bundles and a checksum manifest, without deleting files."""
from pathlib import Path
import zipfile, json, hashlib, ast, argparse
ap=argparse.ArgumentParser();ap.add_argument('--manifest-only',action='store_true');args=ap.parse_args()
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'artifacts').mkdir(exist_ok=True)
files=[]
for folder in ['inputs','experiments','logs']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
# Group at 18 MB uncompressed to keep each independently readable ZIP manageable.
groups=[];current=[];total=0
for p in sorted(files):
    size=p.stat().st_size
    if current and total+size>18_000_000:groups.append(current);current=[];total=0
    current.append(p);total+=size
if current:groups.append(current)
for i,group in ([] if args.manifest_only else enumerate(groups,1)):
    with zipfile.ZipFile(ROOT/'artifacts'/f'part_{i:02d}.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in group:z.write(p,p.relative_to(ROOT).as_posix())
entries=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if rel.parts[0] in ['cache','inputs','experiments','logs','retraining','tmp','.git','docs','output']:continue
    if '__pycache__' in rel.parts or p.name.startswith('upload_'):continue
    if rel.parts[0]=='data' and p.suffix=='.csv':continue
    if rel.as_posix() in ['MANIFEST.json','submissions/reproduced.csv','src/prepare_archive.py','src/build_handbook.py','requirements-pdf.txt']:continue
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
    data=p.read_bytes()
    entries.append({'path':rel.as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                    'git_blob_sha':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()})
(ROOT/'MANIFEST.json').write_text(json.dumps(entries,indent=2))
print(json.dumps({'files':len(entries),'bytes':sum(x['bytes'] for x in entries),'archives':len(groups)},indent=2))
