import json,sys; sys.path.insert(0,'code')
from common import *
from scipy.stats import spearmanr
out={}
S=snapshot_rows()
okA=S[A_FEATS].notna().all(axis=1); okB=S[B_FEATS].notna().all(axis=1)
out['n_rows_snapshots']=len(S)
out['complete_A']=int(okA.sum()); out['complete_B']=int(okB.sum()); out['complete_both']=int((okA&okB).sum())
out['countries_A_ever']=int(S[okA].country.nunique()); out['countries_B_ever']=int(S[okB].country.nunique()); out['countries_both_ever']=int(S[okA&okB].country.nunique())
out['per_year']={int(y):dict(A=int((okA&(S.year==y)).sum()),B=int((okB&(S.year==y)).sum()),both=int((okA&okB&(S.year==y)).sum())) for y in YEARS}
# also A with primary fossil share (conventional-extended)
Aext=A_FEATS+['fossil_primary_sh']
out['complete_Aext']=int(S[Aext].notna().all(axis=1).sum()); out['countries_Aext_ever']=int(S[S[Aext].notna().all(axis=1)].country.nunique())
print(json.dumps(out,indent=1))
C=S[okA&okB].reset_index(drop=True)   # common sample (pooled country-years)
C.to_pickle('results/common_sample.pkl')
res={}
for rep in 'AB':
    m=build(rep,C); res[rep]=m
    ev=m['ev']; print(rep,'feats',len(m['feats']),'npc(parallel)',m['npc'],'EV ratio first3',(ev/ev.sum())[:3].round(3),'cum var kept',m['pca'].explained_variance_ratio_.sum().round(3))
    print(' participation ratio (effective dim)',round(ev.sum()**2/(ev**2).sum(),2))
# cluster selection
tab=[]
for rep in 'AB':
    sc=res[rep]['sc']
    for k in range(2,11):
        km_=km(sc,k); lab=km_.labels_
        tab.append(dict(rep=rep,k=k,sil=silhouette_score(sc,lab),ch=calinski_harabasz_score(sc,lab),db=davies_bouldin_score(sc,lab)))
T=pd.DataFrame(tab); print(T.round(3).to_string()); T.to_csv('results/cluster_selection.csv',index=False)
