"""Rebuild the submitted ensemble from frozen, self-trained predictions."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import runtime
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def read(name):
    with np.load(ROOT / 'inputs' / (name + '.npz')) as z:
        return {k: z[k] for k in z.files}

def main():
    spec = json.loads((ROOT / 'ensemble_spec.json').read_text())
    base = read('own_champion')
    models = [read(n) for n in ['component_lgbm', 'component_realmlp',
              'component_tabm', 'component_tabm3022', 'component_tabm4022',
              'own_teacher', 'realmlp_compact_s33']]
    result = {}
    for part in ['oof', 'test']:
        arrays = [models[0][part], models[1][part],
                  np.mean([m[part] for m in models[2:5]], axis=0),
                  models[5][part], models[6][part]]
        assert all(np.isfinite(a).all() for a in arrays)
        result[part] = np.column_stack([rankdata(a) / len(a) for a in arrays]) @ np.array(list(spec['weights'].values()))
    output = ROOT / 'submissions' / 'reproduced.csv'
    pd.DataFrame({'id': base['test_id'], 'satisfaction': result['test']}).to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert digest == spec['sha256'], (digest, spec['sha256'])
    auc = roc_auc_score(base['target'], result['oof'])
    assert abs(auc - spec['metrics']['auc']) < 1e-12
    report = {'byte_identical': True, 'sha256': digest, 'oof_auc': auc,
              'public_lb_verified': spec['public_lb'], 'test_rows': len(base['test_id'])}
    (ROOT / 'reproduction_metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
