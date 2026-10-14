"""Check all archived file bytes and the final model provenance."""
import runtime
from runtime import ROOT
import json,hashlib,zipfile
checked=0
for path in sorted((ROOT/'artifacts').glob('*.zip')):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None,path
        for info in z.infolist():
            target=(ROOT/info.filename).resolve()
            assert target.is_relative_to(ROOT)
            assert hashlib.sha256(z.read(info)).digest()==hashlib.sha256(target.read_bytes()).digest(),info.filename
            checked+=1
baseline=json.loads((ROOT/'research/baseline_input_hashes.json').read_text())
for name,digest in baseline.items():
    p=ROOT/'inputs'/name
    if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
report={'archive_members_verified_against_local':checked,'baseline_input_hashes_verified':True,
        'external_prediction_files_present':any((ROOT/'inputs').glob('bp_*'))}
assert not report['external_prediction_files_present']
(ROOT/'archive_validation.json').write_text(json.dumps(report,indent=2));print(report)
