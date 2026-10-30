"""Out-of-time transfer when the feature screening itself uses only 2000-2010 (no information from test years)."""
import json,sys; sys.path.insert(0,'code')
from common import *
o=json.load(open('results/p15_extra_controls.json')); feats=o['screen_training_years']
C=pd.read_pickle('results/common_labeled.pkl').reset_index(drop=True)
d=pd.read_pickle('results/panel.pkl'); 
# rows of the common sample that are also complete on the training-selected set
cols=d.set_index(['country','year'])[feats]; Cm=C.join(cols,on=['country','year'],rsuffix='_ts') if False else C.copy()
m=C[['country','year']].merge(d[['country','year']+[f for f in feats if f not in C.columns]],on=['country','year'],how='left')
for f in feats:
    if f not in C.columns: C[f]=m[f].values
ok=C[feats].notna().all(axis=1); Cc=C[ok].reset_index(drop=True); tr=(Cc.year<=2010).values; te=~tr
X=tf(Cc,feats,'B'); lo,hi=np.nanpercentile(X[tr],1,axis=0),np.nanpercentile(X[tr],99,axis=0); Xw=X.clip(lo,hi,axis=1); Z=(Xw-Xw[tr].mean())/Xw[tr].std(ddof=0)
ev,thr,npc=parallel_analysis(Z[tr].values); pca=PCA(n_components=max(npc,2),random_state=SEED).fit(Z[tr].values)
Str,Ste=pca.transform(Z[tr].values),pca.transform(Z[te].values); kk=km(Str,5); pred=kk.predict(Ste); refit=km(Ste,5)
wss=float(((Ste-kk.cluster_centers_[pred])**2).sum())
res=dict(features=feats,n_rows=int(ok.sum()),n_train=int(tr.sum()),n_test=int(te.sum()),npc=int(npc),ARI_vs_test_refit=adjusted_rand_score(refit.labels_,pred),wss_ratio=wss/float(refit.inertia_),sil_transferred=float(silhouette_score(Ste,pred)),sil_refit=float(silhouette_score(Ste,refit.labels_)))
json.dump(res,open('results/p16_transfer_train_selected.json','w'),indent=1,default=float); print(json.dumps(res,indent=1))
