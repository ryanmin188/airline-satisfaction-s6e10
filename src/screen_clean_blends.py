"""Small, predeclared comparison of flatter/sparser self-trained ensembles.
Historical base OOFs are reused: meta CV is diagnostic, not nested base CV.
"""
import runtime
from runtime import ROOT
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
OUT=ROOT;DEST=OUT/'clean_blends';DEST.mkdir(exist_ok=True)
def read(name):
    path=OUT/'inputs'/f'{name}.npz'
    expected=json.loads((OUT/'research/baseline_input_hashes.json').read_text())[path.name]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    with np.load(path) as z:return {k:z[k] for k in z.files}
b=read('own_champion'); y=b['target']; folds=b['fold']; rank=lambda a:rankdata(a)/len(a)
names=['lgbm','realmlp','tabm','teacher'];d=[read('component_lgbm'),read('component_realmlp'),None,read('own_teacher')]
tms=[read('component_'+n) for n in ['tabm','tabm3022','tabm4022']]
d[2]={p:np.mean([m[p] for m in tms],axis=0) for p in ['oof','test']}
base=.7*b['oof']+.3*rank(d[3]['oof'])
baset=.7*b['test']+.3*rank(d[3]['test'])
weights={'current':[.224,.336,.14,.3],'four_round':[.2,.4,.1,.3],
 'three_no_old_lgb':[0,.5,.1,.4],'three_no_tabm':[.2,.4,0,.4],
 'two_mlp40_teacher60':[0,.4,0,.6],'two_equal':[0,.5,0,.5],
 'two_mlp60_teacher40':[0,.6,0,.4],'two_mlp70_teacher30':[0,.7,0,.3]}
rows=[];predictions={};configs={}
def add(name,p,q,config):
    assert np.isfinite(p).all() and np.isfinite(q).all()
    row={'candidate':name,'auc':roc_auc_score(y,p),'selection_auc':roc_auc_score(y[folds<2],p[folds<2]),
         'confirmation_auc':roc_auc_score(y[folds>=2],p[folds>=2]),'components':config['components']}
    for f in range(5):
        ix=folds==f;row[f'fold_{f+1}_gain']=roc_auc_score(y[ix],p[ix])-roc_auc_score(y[ix],base[ix])
    rows.append(row);predictions[name]=(p,q);configs[name]=config
add('verified_096105',base,baset,{'components':4,'kind':'rank','weights':weights['current']})
for transform,fun in [('rank',rank),('probability',lambda a:a),('logit',lambda a:logit(np.clip(a,1e-6,1-1e-6)))]:
    P=np.column_stack([fun(m['oof']) for m in d]);T=np.column_stack([fun(m['test']) for m in d])
    for key,w in weights.items():
        p=P@w;q=T@w
        if transform=='logit':p=expit(p);q=expit(q)
        add(transform+'_'+key,p,q,{'components':int(np.count_nonzero(w)),'kind':transform,'weights':w})
# A single fixed L2 regularization setting, evaluated out of meta-fold.
for cols in [[0,1,2,3],[0,1,3],[1,3]]:
    P=np.column_stack([logit(np.clip(d[j]['oof'],1e-6,1-1e-6)) for j in cols])
    T=np.column_stack([logit(np.clip(d[j]['test'],1e-6,1-1e-6)) for j in cols])
    p=np.zeros(len(y));coefs=[]
    for f in range(5):
        tr=folds!=f;va=~tr
        m=LogisticRegression(C=.1,max_iter=500).fit(P[tr],y[tr])
        p[va]=m.predict_proba(P[va])[:,1];coefs.append(m.coef_[0].tolist())
    m=LogisticRegression(C=.1,max_iter=500).fit(P,y)
    add('logistic_'+str(len(cols)),p,m.predict_proba(T)[:,1],{'components':len(cols),'kind':'logistic_logit','columns':[names[i] for i in cols],
        'C':.1,'coef':m.coef_[0].tolist(),'intercept':m.intercept_[0],'fold_coefficients':coefs})
df=pd.DataFrame(rows).sort_values('selection_auc',ascending=False);df.to_csv(DEST/'screen.csv',index=False)
pd.DataFrame(np.corrcoef(np.column_stack([rank(m['oof']) for m in d]).T),index=names,columns=names).to_csv(DEST/'rank_correlations.csv')
# Select on historical folds 1/2; report (not refit on) remaining folds.
chosen=df.iloc[0].candidate
for name in dict.fromkeys([chosen,df[df.components<=3].iloc[0].candidate,df[df.components<=2].iloc[0].candidate]):
    p,q=predictions[name]
    pd.DataFrame({'id':b['test_id'],'satisfaction':q}).to_csv(DEST/(name+'.csv'),index=False)
    np.savez_compressed(DEST/(name+'_oof.npz'),oof=p)
    meta={'config':configs[name],'metrics':df.set_index('candidate').loc[name].to_dict(),'public_lb':None,
          'selection':'historical folds 1/2; diagnostic folds 3/4/5; no untouched/nested holdout','external_predictions_used':False}
    (DEST/(name+'.json')).write_text(json.dumps(meta,indent=2))
print(df.to_string(index=False),flush=True)
print('SELECTED',chosen,flush=True)
