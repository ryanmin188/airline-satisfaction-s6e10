"""Own training: original statistics, numeric digits, route residuals, inner TE.
Ideas: busyaprime/s6e10-route-or-distance-both-lb-0-96169 (Apache-2.0).
Independent implementation; no external predictions are used.
"""
import runtime
from runtime import ROOT
import argparse,gc,json,time,hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.spatial import cKDTree
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import TargetEncoder

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',type=int,nargs='+',default=[1,2,3,4,5]);ap.add_argument('--model',choices=['lgb','cat','xgb'],default='lgb');args=ap.parse_args()
    out=ROOT/'experiments'/('enhanced_'+args.model);out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(ROOT/'data/train.csv');te=pd.read_csv(ROOT/'data/test.csv');orig=pd.read_csv(ROOT/'data/original.csv')
    y=tr.satisfaction.to_numpy(np.int8);feats=[c for c in tr if c not in ['id','satisfaction']]
    full=pd.concat([tr[feats],te[feats]],ignore_index=True)
    cats=list(full.select_dtypes(exclude='number').columns)
    continuous=['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']
    for c in cats:
        levels=sorted(set(full[c].astype(str))|set(orig[c].astype(str)))
        for f in [full,orig]:f[c]=pd.Categorical(f[c],categories=levels).codes.astype('int16')
    for c in set(feats)-set(cats):
        for f in [full,orig]:f[c]=f[c].fillna(-1).astype('float32')
    fixed=[c for c in feats if c not in continuous]
    oh=pd.util.hash_pandas_object(orig[fixed],index=False).to_numpy();fh=pd.util.hash_pandas_object(full[fixed],index=False).to_numpy()
    og=pd.Series(np.arange(len(orig))).groupby(oh).indices;fg=pd.Series(np.arange(len(full))).groupby(fh).indices
    keep=np.ones(len(orig),bool);scale=np.array([1,10,5,5]);oa=orig[continuous].to_numpy()/scale;fa=full[continuous].to_numpy()/scale
    for key,oi in og.items():
        fi=fg.get(key)
        if fi is None:continue
        distances,_=cKDTree(fa[fi]).query(oa[oi],k=1,p=np.inf,distance_upper_bound=1.000001)
        keep[oi[np.isfinite(distances)]]=False
    print('Original exclusion',int((~keep).sum()),flush=True)
    clean=orig.loc[keep].copy();g=clean.satisfaction.mean()
    shared=ROOT/'experiments/original_teacher';shared.mkdir(exist_ok=True)
    teacher_path=shared/'predictions.npy'
    if not teacher_path.exists():
        teacher=xgb.XGBClassifier(n_estimators=600,learning_rate=.05,max_depth=8,subsample=.8,colsample_bytree=.5,tree_method='hist',n_jobs=4,random_state=0)
        teacher.fit(clean[feats],clean.satisfaction.astype(np.int8));teacher.save_model(shared/'model.ubj')
        np.save(teacher_path,teacher.predict_proba(full[feats])[:,1]);del teacher;gc.collect()
    X=full.copy();X['teacher_logit']=logit(np.clip(np.load(teacher_path),1e-6,1-1e-6)).astype('float32')
    X['arrival_missing']=(full['Arrival Delay in Minutes']==-1).astype('int8')
    for c in feats:
        stats=clean.groupby(c).satisfaction.agg(['mean','count'])
        # Unsmoothed and lightly smoothed original statistics supply distinct support levels.
        X['original_'+c]=full[c].map(stats['mean']).fillna(g).astype('float32')
    route=full.groupby('Flight Distance')
    X['route_count']=full['Flight Distance'].map(full['Flight Distance'].value_counts()).astype('float32')
    for c in feats:
        if c=='Flight Distance':continue
        mean=route[c].transform('mean').astype('float32');X['route_mean_'+c]=mean;X['route_residual_'+c]=(full[c]-mean).astype('float32')
    for c in continuous:
        for k in range(-4,4):X[f'digit_{c}_{k}']=(full[c].fillna(0)//(10.**k)%10).astype('int8')
    for c in list(X):
        if c.startswith(('original_','teacher_')):continue
        f=X[c].map(X[c].value_counts(normalize=True)).astype('float32')
        X['frequency_'+c]=f;X['rarity_'+c]=(-np.log(f.clip(lower=1/len(X)))).astype('float32')
    keys={c:full[c].astype(str) for c in feats}
    pairs=[('Class','Type of Travel'),('Customer Type','Type of Travel'),('Class','Customer Type'),('Gender','Class'),('Inflight wifi service','Online boarding'),('Inflight wifi service','Type of Travel'),('Online boarding','Type of Travel'),('Seat comfort','Inflight entertainment'),('Class','Inflight wifi service'),('Class','Online boarding')]
    for a,b in pairs:keys[a+'__'+b]=full[a].astype(str)+'|'+full[b].astype(str)
    K=pd.DataFrame(keys);X=X.copy()
    config={'model':args.model,'base_features':X.shape[1],'target_encoding_features':len(K.columns),'split_seed':42,'original_rows':len(orig),'original_excluded':int((~keep).sum()),'original_sha256':hashlib.sha256((ROOT/'data/original.csv').read_bytes()).hexdigest(),'early_stopping':'inner 8% holdout; refit outer training only','source':'https://www.kaggle.com/code/busyaprime/s6e10-route-or-distance-both-lb-0-96169'}
    (out/'config.json').write_text(json.dumps(config,indent=2));print(config,flush=True)
    def matrices(fit,others,seed):
        enc=TargetEncoder(target_type='binary',smooth='auto',cv=StratifiedKFold(5,shuffle=True,random_state=seed))
        vals=[enc.fit_transform(K.iloc[fit],y[fit])]+[enc.transform(K.iloc[ix]) for ix in others]
        frames=[]
        for ix,z in zip([fit]+others,vals):
            f=pd.concat([X.iloc[ix].reset_index(drop=True),pd.DataFrame(z.astype('float32'),columns=['TE_'+c for c in K])],axis=1)
            for c in cats:f[c]=f[c].astype('category') if args.model!='cat' else f[c].astype(str)
            f.columns=[c.replace(' ','_').replace('/','_') for c in f.columns];frames.append(f)
        return frames
    params=dict(objective='binary',learning_rate=.02,max_depth=8,num_leaves=127,colsample_bytree=.5,subsample=.9,subsample_freq=1,min_child_samples=20,reg_lambda=5.,reg_alpha=.05,n_jobs=6,random_state=42,verbosity=-1)
    def model(n,early=False):
        if args.model=='lgb':return lgb.LGBMClassifier(n_estimators=n,**params)
        if args.model=='xgb':return xgb.XGBClassifier(n_estimators=n,learning_rate=.03,max_depth=8,subsample=.9,colsample_bytree=.5,min_child_weight=5,reg_lambda=5,reg_alpha=.05,tree_method='hist',device='cuda',enable_categorical=True,max_cat_to_onehot=8,eval_metric='auc',early_stopping_rounds=200 if early else None,n_jobs=6,random_state=42)
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=n,learning_rate=.05,depth=8,l2_leaf_reg=5,border_count=254,eval_metric='AUC',task_type='GPU',devices='0',cat_features=[c.replace(' ','_') for c in cats],random_seed=42,verbose=False,allow_writing_files=False)
    for f,(fit,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y),1):
        if f not in args.folds:continue
        checkpoint=out/f'fold_{f}.npz'
        if checkpoint.exists():continue
        started=time.time();inner,es=train_test_split(fit,test_size=.08,stratify=y[fit],random_state=42)
        a,b=matrices(inner,[es],42+f);m=model(6000,True)
        if args.model=='lgb':m.fit(a,y[inner],eval_set=[(b,y[es])],eval_metric='auc',callbacks=[lgb.early_stopping(200,verbose=False)]);n=m.best_iteration_
        elif args.model=='xgb':m.fit(a,y[inner],eval_set=[(b,y[es])],verbose=False);n=m.best_iteration+1
        else:m.fit(a,y[inner],eval_set=(b,y[es]),early_stopping_rounds=200,use_best_model=True);n=m.get_best_iteration()+1
        print('fold',f,'iterations',n,flush=True);del a,b,m;gc.collect()
        a,b,c=matrices(fit,[va,np.arange(len(tr),len(full))],42+f);m=model(n);m.fit(a,y[fit])
        p=m.predict_proba(b)[:,1];q=m.predict_proba(c)[:,1]
        np.savez_compressed(checkpoint,valid_idx=va,valid_pred=p,test_pred=q)
        if args.model=='lgb':m.booster_.save_model(str(out/f'fold_{f}.txt'))
        else:m.save_model(str(out/f'fold_{f}.model'))
        result={'fold':f,'auc':roc_auc_score(y[va],p),'iterations':n,'seconds':time.time()-started}
        (out/f'fold_{f}.json').write_text(json.dumps(result,indent=2));print(result,flush=True);del a,b,c,m;gc.collect()
    if all((out/f'fold_{f}.npz').exists() for f in range(1,6)):
        oof=np.empty(len(y));test=np.zeros(len(te));fold=np.empty(len(y),np.int8)
        for f in range(1,6):
            with np.load(out/f'fold_{f}.npz') as z:oof[z['valid_idx']]=z['valid_pred'];test+=z['test_pred']/5;fold[z['valid_idx']]=f-1
        pd.DataFrame({'id':tr.id,'satisfaction':y,'fold':fold,'prediction':oof}).to_csv(out/'oof.csv',index=False)
        pd.DataFrame({'id':te.id,'prediction':test}).to_csv(out/'test.csv',index=False)
        (out/'summary.json').write_text(json.dumps({'auc':roc_auc_score(y,oof)},indent=2));print('OOF',roc_auc_score(y,oof),flush=True)
if __name__=='__main__':main()
