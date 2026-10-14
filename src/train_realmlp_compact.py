"""Train compact RealMLP locally; public configurations are method references only.
Reference: https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit
Raw numericals remain continuous and categorical; drop two high-cardinality
interaction categories after inner cross-fitted target encoding.
"""
import runtime
from runtime import ROOT
import argparse,gc,json,time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',type=int,nargs='+',default=[1,2,3,4,5]);ap.add_argument('--seed',type=int,default=33);args=ap.parse_args()
    out=ROOT/'experiments'/f'realmlp_compact_s{args.seed}';out.mkdir(exist_ok=True,parents=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    params=dict(random_state=args.seed,verbosity=0,val_metric_name='1-auc_ovr',n_ens=8,n_epochs=3,stop_epoch=3,batch_size=256,use_early_stopping=False,lr=.053,wd=.015,sq_mom=.988,lr_sched='flat_anneal',wd_sched='cos_log_15',first_layer_lr_factor=.25,embedding_size=5,max_one_hot_cat_size=18,hidden_sizes=[512,256,128],act='silu',p_drop=.05,p_drop_sched='invsqrtp1e-3',plr_hidden_1=16,plr_hidden_2=8,plr_act_name='gelu',plr_lr_factor=.1151,plr_sigma=2.33,ls_eps=.01,ls_eps_sched='sqrt_cos',add_front_scale=False,bias_init_mode='neg-uniform-dynamic-2',tfms=['one_hot','median_center','robust_scale','smooth_clip','embedding','l2_normalize'])
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');y=train.satisfaction.to_numpy(np.int8)
    raw=train.drop(columns=['id','satisfaction']);rt=test.drop(columns=['id']);cats=list(raw.select_dtypes(exclude='number').columns);nums=[c for c in raw if c not in cats]
    combos=[('Class','Type of Travel','Gender'),('Class','Type of Travel','Age'),('Flight Distance','Departure/Arrival time convenient')]
    def features(df):
        x=df.copy();names=[]
        for c in cats:x[c]=x[c].fillna('missing').astype(str)
        for c in nums:x[c]=x[c].fillna(0);x[c+'_cat']=x[c].astype(str)
        for cols in combos:
            name='__'.join(cols);names.append(name)
            value=x[cols[0]].astype(str)
            for col in cols[1:]:value=value+'|'+x[col].astype(str)
            x[name]=value
        return x,names
    X,combo=features(raw);Xt,_=features(rt);modelcats=cats+[c+'_cat' for c in nums]+[combo[0]]
    config={'params':params,'split_seed':42,'feature_policy':'outer-training counts/category vocab; inner-fivefold TE; remove 2 interaction categories','torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y),1):
        if f not in args.folds or (out/f'fold_{f}.npz').exists():continue
        start=time.time();a=X.iloc[tr].copy();b=X.iloc[va].copy();c=Xt.copy()
        for col in cats+['Leg room service']:
            counts=a[col].value_counts()
            for frame in [a,b,c]:frame['count_'+col]=frame[col].map(counts).fillna(0).astype('float32')
        enc=TargetEncoder(target_type='binary',smooth='auto',cv=5,shuffle=True,random_state=42+f)
        values=[enc.fit_transform(a[combo],y[tr]),enc.transform(b[combo]),enc.transform(c[combo])]
        for frame,z in zip([a,b,c],values):
            for j,col in enumerate(combo):frame['TE_'+col]=z[:,j].astype('float32')
            frame.drop(columns=combo[1:],inplace=True)
        for col in modelcats:
            levels=list(pd.unique(a[col].astype(str)))+['__unseen__'];dtype=pd.CategoricalDtype(categories=levels)
            for frame in [a,b,c]:frame[col]=frame[col].astype(str).where(frame[col].astype(str).isin(levels),'__unseen__').astype(dtype)
        for frame in [a,b,c]:
            for col in frame.columns.difference(modelcats):frame[col]=frame[col].astype('float32')
        print('fold',f,'features',a.shape[1],flush=True)
        m=RealMLP_TD_Classifier(device='cuda',**(params|{'random_state':args.seed+f}))
        m.fit(a,y[tr],b,y[va],cat_col_names=modelcats)
        p=m.predict_proba(b)[:,1];q=m.predict_proba(c)[:,1]
        np.savez_compressed(out/f'fold_{f}.npz',valid_idx=va,valid_pred=p,test_pred=q)
        metric={'fold':f,'auc':roc_auc_score(y[va],p),'seconds':time.time()-start}
        (out/f'fold_{f}.json').write_text(json.dumps(metric,indent=2));print(metric,flush=True)
        del m,a,b,c;gc.collect();torch.cuda.empty_cache()
    if all((out/f'fold_{f}.npz').exists() for f in range(1,6)):
        p=np.zeros(len(y));q=np.zeros(len(test));fold=np.empty(len(y),np.int8)
        for f in range(1,6):
            with np.load(out/f'fold_{f}.npz') as z:p[z['valid_idx']]=z['valid_pred'];q+=z['test_pred']/5;fold[z['valid_idx']]=f-1
        pd.DataFrame({'id':train.id,'satisfaction':y,'fold':fold,'prediction':p}).to_csv(out/'oof.csv',index=False)
        pd.DataFrame({'id':test.id,'prediction':q}).to_csv(out/'test.csv',index=False)
        (out/'summary.json').write_text(json.dumps({'auc':roc_auc_score(y,p)},indent=2));print('OOF',roc_auc_score(y,p),flush=True)
if __name__=='__main__':main()
