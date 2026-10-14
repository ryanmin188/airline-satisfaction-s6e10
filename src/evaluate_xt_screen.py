"""Two-fold XT screening only; scores cannot be called full OOF AUC."""
import runtime
from runtime import ROOT
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
out=ROOT;xp=out/'experiments/teacher_lgbm_xt'
with np.load(out/'inputs/own_champion.npz') as z:b={k:z[k] for k in z.files}
components={}
for key,file in [('lgb','component_lgbm'),('mlp','component_realmlp'),('teacher','own_teacher')]:
    with np.load(out/'inputs'/f'{file}.npz') as z:components[key]=z['oof']
tabms=[]
for n in ['tabm','tabm3022','tabm4022']:
    with np.load(out/'inputs'/f'component_{n}.npz') as z:tabms.append(z['oof'])
components['tabm']=np.mean(tabms,axis=0)
rank=lambda a:rankdata(a)/len(a)
rows=[]
for f in [1,2]:
    if not (xp/f'fold_{f}.npz').exists():continue
    with np.load(xp/f'fold_{f}.npz') as z:ix=z['valid_idx'];p=z['valid_pred']
    y=b['target'][ix];r={k:rank(v[ix]) for k,v in components.items()}
    # Re-rank every member within this fold for a fair partial-run comparison.
    base=.224*r['lgb']+.336*r['mlp']+.14*r['tabm']+.3*r['teacher']
    ba=roc_auc_score(y,base)
    for w in [0.,.1,.2,.3]:
        pred=(1-w)*base+w*rank(p)
        rows.append({'fold':f,'kind':'add_xt','weight':w,'single_xt_auc':roc_auc_score(y,p),'baseline_auc':ba,'blend_auc':roc_auc_score(y,pred),'gain':roc_auc_score(y,pred)-ba})
    # Replacement keeps the ensemble at four configuration lines.
    pred=.224*r['lgb']+.336*r['mlp']+.14*r['tabm']+.3*rank(p)
    rows.append({'fold':f,'kind':'replace_teacher','weight':.3,'single_xt_auc':roc_auc_score(y,p),'baseline_auc':ba,'blend_auc':roc_auc_score(y,pred),'gain':roc_auc_score(y,pred)-ba})
df=pd.DataFrame(rows);df.to_csv(xp/'blend_screen.csv',index=False);print(df.to_string(index=False))
