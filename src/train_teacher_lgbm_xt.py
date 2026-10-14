"""Original-data teacher + fold-safe exact-value TE, trained locally.

Motivation: https://www.kaggle.com/code/sachith7/s6e10-what-each-step-was-worth
Independent implementation. Original data: arseniyshutko/binary-aviation-satisfaction-129k.
Competition-near-duplicate original rows are excluded before training the teacher.
"""
import runtime
from runtime import ROOT
import argparse, gc, hashlib, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

OUT=ROOT/'experiments/teacher_lgbm_xt'
OUT.mkdir(parents=True,exist_ok=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',type=int,nargs='+',default=[1,2,3,4,5]);args=ap.parse_args()
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    orig=pd.read_csv(ROOT/'data/original.csv')
    y=train.satisfaction.to_numpy(np.int8); feats=[c for c in train if c not in ['id','satisfaction']]
    full=pd.concat([train[feats],test[feats]],ignore_index=True)
    cats=list(full.select_dtypes(include=['str','object','category']).columns)
    nums=[c for c in feats if c not in cats]
    for c in cats:
        levels=sorted(set(full[c].astype(str))|set(orig[c].astype(str)))
        for frame in [full,orig]: frame[c]=pd.Categorical(frame[c],categories=levels).codes.astype('int16')
    for c in nums:
        for frame in [full,orig]: frame[c]=frame[c].fillna(-1).astype('float32')
    teacherfile=OUT/'teacher_features.npy'
    if not teacherfile.exists():
        varying=['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']
        fixed=[c for c in feats if c not in varying]
        # Stable hashes of all ratings/categories, then Chebyshev radius check on continuous values.
        oh=pd.util.hash_pandas_object(orig[fixed],index=False).to_numpy()
        fh=pd.util.hash_pandas_object(full[fixed],index=False).to_numpy()
        og=pd.Series(np.arange(len(orig))).groupby(oh).indices
        fg=pd.Series(np.arange(len(full))).groupby(fh).indices
        keep=np.ones(len(orig),bool); scales=np.array([1,10,5,5])
        oa=orig[varying].to_numpy()/scales;fa=full[varying].to_numpy()/scales
        for key,oi in og.items():
            fi=fg.get(key)
            if fi is None: continue
            distances,_=cKDTree(fa[fi]).query(oa[oi],k=1,p=np.inf,distance_upper_bound=1.000001)
            keep[oi[np.isfinite(distances)]]=False
        # Exclusion does not inspect competition labels.
        audit={'original_rows':len(orig),'excluded_near_duplicates':int((~keep).sum()),'remaining':int(keep.sum()),
               'match':'all ratings/categories identical, abs age<=1, distance<=10, each delay<=5; missing delay filled -1',
               'source':'https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k',
               'source_sha256':hashlib.sha256((ROOT/'data/original.csv').read_bytes()).hexdigest()}
        (OUT/'teacher_audit.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)
        m=xgb.XGBClassifier(n_estimators=600,learning_rate=.05,max_depth=8,subsample=.8,colsample_bytree=.5,tree_method='hist',n_jobs=4,random_state=0)
        m.fit(orig.loc[keep,feats],orig.loc[keep,'satisfaction'].astype(np.int8))
        teacher=logit(np.clip(m.predict_proba(full[feats])[:,1],1e-6,1-1e-6)).astype('float32')
        np.save(teacherfile,teacher);m.save_model(OUT/'teacher.ubj');del m;gc.collect()
    teacher=np.load(teacherfile)
    print('Teacher AUC',roc_auc_score(y,teacher[:len(y)]),flush=True)
    X=full.copy();X['teacher_logit']=teacher
    X['arrival_missing']=(full['Arrival Delay in Minutes']==-1).astype('int8')
    groups=full.groupby('Flight Distance')
    for c in feats:
        if c!='Flight Distance': X['route_'+c]=groups[c].transform('mean').astype('float32')
    for c in nums: X['freq_'+c]=full[c].map(full[c].value_counts()).astype('float32')
    keys={c:full[c].astype(str) for c in nums+cats}
    for cols in [('Flight Distance','Age'),('Flight Distance','Class'),('Flight Distance','Type of Travel'),('Age','Class','Type of Travel','Customer Type'),('Class','Type of Travel','Gender')]:
        col='|'.join(cols);s=full[cols[0]].astype(str)
        for c in cols[1:]:s=s+'|'+full[c].astype(str)
        keys[col]=s
    K=pd.DataFrame(keys)
    baseparams=dict(objective='binary',learning_rate=.03,num_leaves=31,extra_trees=True,min_child_samples=60,colsample_bytree=.4,subsample=.8,subsample_freq=1,reg_alpha=.3,reg_lambda=1.,n_jobs=6,verbosity=-1,random_state=42)
    testidx=np.arange(len(y),len(full))
    def matrices(fitidx, others, seed):
        enc=TargetEncoder(target_type='binary',smooth=20.,cv=StratifiedKFold(5,shuffle=True,random_state=seed))
        vals=[enc.fit_transform(K.iloc[fitidx],y[fitidx])]+[enc.transform(K.iloc[i]) for i in others]
        frames=[]
        for i,z in zip([fitidx]+others,vals):
            frame=X.iloc[i].reset_index(drop=True).copy()
            for j,name in enumerate(K):frame['TE_'+name]=z[:,j].astype('float32')
            # Integer category values are treated as native LightGBM categoricals.
            for name in cats:frame[name]=frame[name].astype('category')
            frame.columns=[c.replace(' ','_').replace('/','_').replace('|','__') for c in frame.columns]
            frames.append(frame)
        return frames
    for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y),1):
        if f not in args.folds:continue
        ck=OUT/f'fold_{f}.npz'
        if ck.exists():
            with np.load(ck) as z:assert np.array_equal(z['valid_idx'],va)
            continue
        start=time.time();fitidx,esidx=train_test_split(tr,test_size=.08,stratify=y[tr],random_state=42)
        a,b=matrices(fitidx,[esidx],42+f)
        m=lgb.LGBMClassifier(n_estimators=8000,**baseparams)
        m.fit(a,y[fitidx],eval_set=[(b,y[esidx])],eval_metric='auc',callbacks=[lgb.early_stopping(200,verbose=False)])
        n=m.best_iteration_;print('fold',f,'best_iteration',n,flush=True)
        del a,b,m;gc.collect()
        a,b,c=matrices(tr,[va,testidx],42+f)
        m=lgb.LGBMClassifier(n_estimators=n,**baseparams);m.fit(a,y[tr])
        pv=m.predict_proba(b)[:,1];pt=m.predict_proba(c)[:,1]
        np.savez_compressed(ck,valid_idx=va,valid_pred=pv,test_pred=pt)
        m.booster_.save_model(str(OUT/f'fold_{f}.txt'))
        metric={'fold':f,'auc':roc_auc_score(y[va],pv),'iterations':n,'seconds':time.time()-start}
        (OUT/f'fold_{f}.json').write_text(json.dumps(metric,indent=2));print(metric,flush=True)
        del a,b,c,m;gc.collect()
    if all((OUT/f'fold_{f}.npz').exists() for f in range(1,6)):
        oof=np.full(len(y),np.nan);pred=np.zeros(len(test));folds=np.zeros(len(y),np.int8)
        for f in range(1,6):
            with np.load(OUT/f'fold_{f}.npz') as z:oof[z['valid_idx']]=z['valid_pred'];pred+=z['test_pred']/5;folds[z['valid_idx']]=f
        assert np.isfinite(oof).all()
        pd.DataFrame({'id':train.id,'satisfaction':y,'fold':folds,'prediction':oof}).to_csv(OUT/'oof.csv',index=False)
        pd.DataFrame({'id':test.id,'prediction':pred}).to_csv(OUT/'test.csv',index=False)
        (OUT/'summary.json').write_text(json.dumps({'auc':roc_auc_score(y,oof),'public_lb':None},indent=2))
        print('OOF',roc_auc_score(y,oof),flush=True)

if __name__=='__main__':main()
