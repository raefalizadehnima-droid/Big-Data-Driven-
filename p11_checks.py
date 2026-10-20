"""Corrections: rank-displacement recount, sample-flow, dataset fingerprint, leakage-free temporal transfer."""
import json,sys,hashlib,os; sys.path.insert(0,'code')
from common import *
from scipy.stats import spearmanr, kendalltau
out={}
# --- dataset fingerprint ---
p='/mnt/user-data/uploads/owid_energy.csv'; raw=pd.read_csv(p)
out['file']=dict(sha256=hashlib.sha256(open(p,'rb').read()).hexdigest(),bytes=os.path.getsize(p),rows=raw.shape[0],cols=raw.shape[1],max_year=int(raw.year.max()),last_modified_utc=pd.Timestamp(os.path.getmtime(p),unit='s').isoformat())
# --- sample flow ---
d=pd.read_pickle('results/panel.pkl'); S=snapshot_rows()
f={}
f['iso_countries']=int(d.country.nunique()); f['panel_country_years']=len(d)
f['snapshot_rows']=len(S); f['snapshot_possible']=len(YEARS)*f['iso_countries']
okA=S[A_FEATS].notna().all(axis=1); okB=S[B_FEATS].notna().all(axis=1)
f['A_complete_rows']=int(okA.sum()); f['B_complete_rows']=int(okB.sum()); f['both_rows']=int((okA&okB).sum()); f['both_countries']=int(S[okA&okB].country.nunique())
C=pd.read_pickle('results/common_labeled.pkl'); S22=C[C.year==2022]
f['common_2022_rows']=len(S22)
need=['fossil_sh','energy_int','total_energy_per_capita_kwh','carbon_int','elec_div','import_exp']
miss=S22[S22[need].isna().any(axis=1)]
f['objective_complete_2022']=len(S22)-len(miss)
f['dropped_2022']=[dict(country=r.country,missing=[c for c in need if pd.isna(r[c])]) for _,r in miss.iterrows()]
per={}
for y in YEARS: per[y]=dict(possible=int(d[d.year==y].country.nunique()),A=int((okA&(S.year==y)).sum()),B=int((okB&(S.year==y)).sum()),both=int((okA&okB&(S.year==y)).sum()))
f['per_year']=per
out['flow']=f
# --- rank displacement recount from the rank vectors ---
R=pd.read_pickle('results/snapshot2022_ranked.pkl'); n=len(R); disp=(R.rankA-R.rankB).abs().values
D={'n':n}
for k in (10,16,20,32,40): D[f'ge_{k}_places']=dict(count=int((disp>=k).sum()),share=float((disp>=k).mean()),pct_of_list=round(100*k/n,1))
q=lambda r:np.minimum(((r-1)*5//n).astype(int),4); dq=np.abs(q(R.rankA.values)-q(R.rankB.values))
D['quintile_changed']=float((dq>=1).mean()); D['quintile_ge2']=float((dq>=2).mean()); D['quintile_ge2_count']=int((dq>=2).sum())
tau=kendalltau(R.rankA,R.rankB)[0]; D['tau']=float(tau); D['pair_reversal']=float((1-tau)/2)
rng=np.random.default_rng(SEED); b=[]
for _ in range(2000):
    i=rng.choice(n,n,replace=True); b.append((1-kendalltau(R.rankA.values[i],R.rankB.values[i])[0])/2)
D['pair_reversal_ci']=np.percentile(b,[2.5,97.5]).tolist()
D['ties_A']=int(R.rankA.duplicated().sum()); D['ties_B']=int(R.rankB.duplicated().sum())
D['mean']=float(disp.mean()); D['median']=float(np.median(disp)); D['p90']=float(np.percentile(disp,90)); D['max']=int(disp.max())
out['rank_disp']=D
# --- leakage-free temporal transfer ---
K={'A':4,'B':5}; T={}
for rep in 'AB':
    feats=A_FEATS if rep=='A' else B_FEATS; X=tf(C,feats,'A' if rep=='A' else 'B'); tr=(C.year<=2010).values; te=~tr
    # preprocessing parameters estimated on training years only
    lo,hi=np.nanpercentile(X[tr],1,axis=0),np.nanpercentile(X[tr],99,axis=0); Xw=X.clip(lo,hi,axis=1); mu,sd=Xw[tr].mean(),Xw[tr].std(ddof=0); Z=(Xw-mu)/sd
    ev,thr,npc=parallel_analysis(Z[tr].values); pca=PCA(n_components=max(npc,2),random_state=SEED).fit(Z[tr].values)
    Str,Ste=pca.transform(Z[tr].values),pca.transform(Z[te].values); kk=km(Str,K[rep]); pred=kk.predict(Ste)
    # reference 1: partition from the pooled model (uses all years, so it is NOT leakage-free; kept for comparability)
    pooled=C[f'cl_{rep}'].values
    # reference 2: partition obtained from test years only (own preprocessing)
    Xt=X[te]; lo2,hi2=np.nanpercentile(Xt,1,axis=0),np.nanpercentile(Xt,99,axis=0); Xt=Xt.clip(lo2,hi2,axis=1); Zt=(Xt-Xt.mean())/Xt.std(ddof=0)
    ev2,thr2,npc2=parallel_analysis(Zt.values); p2=PCA(n_components=max(npc2,2),random_state=SEED).fit(Zt.values); St2=p2.transform(Zt.values); k2=km(St2,K[rep])
    # transfer efficiency in the test years' own PCA space: WSS(train centroids) vs WSS(test-refit centroids)
    def wss(S_,cent,lab): return float(((S_-cent[lab])**2).sum())
    # assign test rows to the train centroids expressed in the train PCA space; compute WSS there, and compare with a k-means refit in the same (train) PCA space
    wss_train_cent=wss(Ste,kk.cluster_centers_,pred); refit=km(Ste,K[rep]); wss_refit=float(refit.inertia_)
    T[rep]=dict(n_train=int(tr.sum()),n_test=int(te.sum()),npc_train=int(npc),npc_test_own=int(npc2),k=K[rep],
        ARI_vs_pooled=adjusted_rand_score(pooled[te],pred),ARI_vs_test_refit=adjusted_rand_score(refit.labels_,pred),ARI_vs_test_own_pipeline=adjusted_rand_score(k2.labels_,pred),
        wss_ratio=wss_train_cent/wss_refit,silhouette_test_train_labels=float(silhouette_score(Ste,pred)),silhouette_test_refit=float(silhouette_score(Ste,refit.labels_)),
        in_sample_ARI_vs_pooled=adjusted_rand_score(pooled[tr],kk.labels_),
        params=dict(winsor_lo=[float(x) for x in lo],winsor_hi=[float(x) for x in hi],mean=[float(x) for x in mu],sd=[float(x) for x in sd]))
out['temporal_transfer']=T
json.dump(out,open('results/p11_checks.json','w'),indent=1,default=float)
print(json.dumps({k:(v if k!='temporal_transfer' else {r:{a:b for a,b in T[r].items() if a!='params'} for r in T}) for k,v in out.items()},indent=1,default=float)[:6000])
