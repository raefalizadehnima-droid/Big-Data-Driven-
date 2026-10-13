import json,sys; sys.path.insert(0,'code')
from common import *
from scipy.stats import wilcoxon
C=pd.read_pickle('results/common_labeled.pkl').reset_index(drop=True)
p4=json.load(open('results/p4_results.json')); p5=json.load(open('results/p5_results.json')); p6=json.load(open('results/p6_results.json')); v1=json.load(open('results/p1_verification.json'))
# --- holdout diagnostic saved ---
diag={}
for r,k in (('A',4),('B',5)):
    feats=A_FEATS if r=='A' else B_FEATS; X=tf(C,feats,'A' if r=='A' else 'B')
    def fit(mask):
        a,b=np.nanpercentile(X[mask],1,axis=0),np.nanpercentile(X[mask],99,axis=0); Xw=X.clip(a,b,axis=1); Z=(Xw-Xw[mask].mean())/Xw[mask].std(ddof=0)
        ev,thr,npc=parallel_analysis(Z[mask].values); pca=PCA(n_components=max(npc,2),random_state=SEED).fit(Z[mask].values); return Z,pca,km(pca.transform(Z[mask].values),k)
    tr=(C.year<=2010).values; Z,pca,kk=fit(tr)
    diag[r]=dict(in_sample_ARI=adjusted_rand_score(C[f'cl_{r}'].values[tr],kk.labels_),out_of_time_ARI=adjusted_rand_score(C[f'cl_{r}'].values[~tr],kk.predict(pca.transform(Z[~tr].values))))
json.dump(diag,open('results/p6b_holdout_diag.json','w'),indent=1,default=float); print(diag)
# --- Holm ---
tests=[('H1 Mantel (distance-matrix agreement), 2022',p4['latent']['2022']['mantel_p']),('H1 Procrustes (2-PC configuration), 2022',p4['latent']['2022']['procrustes_p']),('H2 ARI vs chance (A4 vs B5), permutation',p4['agreement']['selected(A4,B5)']['perm_p']),('H4 cluster stability B vs A, paired Wilcoxon',p6['cluster_stability_paired']['wilcoxon_p']),('H4 rank stability B vs A, paired Wilcoxon',p6['rank_stability_subsample']['wilcoxon_p'])]
ps=np.array([t[1] for t in tests]); o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
for rank,i in enumerate(o): run=max(run,(m-rank)*ps[i]); adj[i]=min(1,run)
H=pd.DataFrame({'test':[t[0] for t in tests],'p_raw':ps,'p_holm':adj}); H.to_csv('results/holm.csv',index=False); print(H)
