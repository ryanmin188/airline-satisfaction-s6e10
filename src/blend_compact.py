import runtime
from runtime import ROOT
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
def read(name):
    with np.load(ROOT/'inputs'/f'{name}.npz') as z:return {k:z[k] for k in z.files}
b=read('own_champion');teacher=read('own_teacher');lgb=read('component_lgbm');mlp=read('component_realmlp')
tm=[read('component_'+s) for s in ['tabm','tabm3022','tabm4022']]
newpath=ROOT/'experiments/realmlp_compact_s33';a=pd.read_csv(newpath/'oof.csv').set_index('id').loc[b['id']];t=pd.read_csv(newpath/'test.csv').set_index('id').loc[b['test_id']]
assert np.array_equal(a.satisfaction.to_numpy(),b['target'])
new={'oof':a.prediction.to_numpy(),'test':t.prediction.to_numpy()};np.savez_compressed(ROOT/'inputs/realmlp_compact_s33.npz',**new)
rank=lambda a:rankdata(a)/len(a)
cols=['lgbm','realmlp_v1','tabm_three_seed','teacher_lgbm','realmlp_compact']
P={part:np.column_stack([rank(lgb[part]),rank(mlp[part]),rank(np.mean([m[part] for m in tm],axis=0)),rank(teacher[part]),rank(new[part])]) for part in ['oof','test']}
old=np.array([.224,.336,.14,.3,0]);base=.7*b['oof']+.3*rank(teacher['oof'])
weights={'current':old,'add_compact20':old*.8+np.array([0,0,0,0,.2]),'add_compact30':old*.7+np.array([0,0,0,0,.3]),
 'add_compact40':old*.6+np.array([0,0,0,0,.4]),'replace_half_v1':[.224,.168,.14,.3,.168],
 'replace_v1':[.224,0,.14,.3,.336],'compact_four_40':[.15,0,.1,.35,.4],
 'compact_four_50':[.1,0,.1,.3,.5],'compact_three':[0,0,.15,.4,.45]}
rows=[];ys=b['target'];fold=b['fold']
for name,w in weights.items():
    p=P['oof']@w
    row={'candidate':name,'auc':roc_auc_score(ys,p),'selection_auc':roc_auc_score(ys[fold<2],p[fold<2]),'confirmation_auc':roc_auc_score(ys[fold>=2],p[fold>=2]),'components':int(np.count_nonzero(w))}
    for f in range(5):
        ix=fold==f;row[f'fold_{f+1}_gain']=roc_auc_score(ys[ix],p[ix])-roc_auc_score(ys[ix],base[ix])
    rows.append(row)
df=pd.DataFrame(rows).sort_values('selection_auc',ascending=False);df.to_csv(ROOT/'experiments/compact_blend_screen.csv',index=False)
chosen=df.iloc[0].candidate
for name in dict.fromkeys([chosen,'replace_v1','compact_four_50']):
    w=np.array(weights[name]);q=P['test']@w;p=P['oof']@w
    dest=ROOT/'submissions'/(name+'.csv');pd.DataFrame({'id':b['test_id'],'satisfaction':q}).to_csv(dest,index=False)
    np.savez_compressed(ROOT/'submissions'/(name+'_oof.npz'),oof=p)
    record={'metrics':df.set_index('candidate').loc[name].to_dict(),'weights':dict(zip(cols,w.tolist())),'public_lb':None,'external_predictions_used':False,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'selection':'historical folds 1/2; confirm 3/4/5; not independent nested CV'}
    record_path=ROOT/'submissions'/(name+'.json')
    if record_path.exists():
        previous=json.loads(record_path.read_text())
        if previous.get('sha256')==record['sha256']:
            record['public_lb']=previous.get('public_lb')
    record_path.write_text(json.dumps(record,indent=2))
print(df.to_string(index=False));print('SELECTED',chosen)
