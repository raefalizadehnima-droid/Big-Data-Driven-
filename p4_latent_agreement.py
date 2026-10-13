import json,sys; sys.path.insert(0,'code')
from common import *
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
C=pd.read_pickle('results/common_sample.pkl'); rng=np.random.default_rng(SEED)
M={r:build(r,C) for r in 'AB'}
K={'A':4,'B':5}
lab={r:km(M[r]['sc'],K[r]).labels_ for r in 'AB'}
C['cl_A']=lab['A']; C['cl_B']=lab['B']
for k in (4,5,6):
    C[f'A_k{k}']=km(M['A']['sc'],k).labels_; C[f'B_k{k}']=km(M['B']['sc'],k).labels_
C.to_pickle('results/common_labeled.pkl')
res={}
# ---------- H1: latent space comparison (2022 snapshot, own standardisation) ----------
L={}
for y in (2022,2000):
    S=C[C.year==y].reset_index(drop=True)
    mA,mB=build('A',S),build('B',S)
    dA=pdist(mA['Z'].values); dB=pdist(mB['Z'].values)
    rho=spearmanr(dA,dB)[0]
    # Mantel permutation (row-permute B distance matrix)
    DB=squareform(dB); n=len(S); cnt=0; NP=2000
    for _ in range(NP):
        p=rng.permutation(n); r=spearmanr(dA,squareform(DB[np.ix_(p,p)]))[0]
        cnt+= r>=rho
    pm=(cnt+1)/(NP+1)
    # Procrustes on first 2 PCs
    _,_,disp=procrustes(mA['sc'][:,:2],mB['sc'][:,:2]); cnt=0
    for _ in range(NP):
        p=rng.permutation(n); _,_,dd=procrustes(mA['sc'][:,:2],mB['sc'][p,:2]); cnt+= dd<=disp
    pp=(cnt+1)/(NP+1)
    # kNN overlap k=5 and noise floor via random split of B features
    def knn(Z,k=5):
        D=squareform(pdist(Z)); np.fill_diagonal(D,np.inf); return np.argsort(D,axis=1)[:,:k]
    def jac(a,b): return np.mean([len(set(x)&set(z))/len(set(x)|set(z)) for x,z in zip(a,b)])
    kA,kB=knn(mA['Z'].values),knn(mB['Z'].values); j_AB=jac(kA,kB)
    # null: random neighbour sets
    jn=[jac(kA,knn(rng.normal(size=(n,3)))) for _ in range(100)]
    # noise floor: B features split in halves
    Bz=mB['Z'].values; fl=[]
    for _ in range(100):
        perm=rng.permutation(Bz.shape[1]); h1,h2=perm[:6],perm[6:]
        fl.append(jac(knn(Bz[:,h1]),knn(Bz[:,h2])))
    # A split can't be done (3 feats); use A features vs B halves
    jAh=[]
    for _ in range(100):
        perm=rng.permutation(Bz.shape[1]); jAh.append(jac(kA,knn(Bz[:,perm[:6]])))
    L[y]=dict(n=n,mantel_rho=rho,mantel_p=pm,procrustes_disp=disp,procrustes_p=pp,knn_jaccard_AB=j_AB,knn_null_mean=float(np.mean(jn)),knn_Bhalf_vs_Bhalf_mean=float(np.mean(fl)),knn_Bhalf_vs_Bhalf_sd=float(np.std(fl)),knn_A_vs_Bhalf=float(np.mean(jAh)))
res['latent']=L
# loadings
for r in 'AB':
    pca=M[r]['pca']; ld=pd.DataFrame(pca.components_.T,index=M[r]['feats'],columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    ld.to_csv(f'results/loadings_{r}.csv'); print(r,'EVR',pca.explained_variance_ratio_.round(3)); print(ld.round(2).to_string())
    res[f'evr_{r}']=pca.explained_variance_ratio_.tolist(); res[f'npc_parallel_{r}']=int(M[r]['npc'])
    res[f'pa_eigs_{r}']=dict(ev=M[r]['ev'].tolist(),thr=M[r]['thr'].tolist())
# ---------- H2: classification agreement ----------
def agree(a,b): return dict(ARI=adjusted_rand_score(a,b),AMI=adjusted_mutual_info_score(a,b),VI=vi(a,b))
AG={}
cs=C.country.values; ucs=np.unique(cs); idx={c:np.where(cs==c)[0] for c in ucs}
def boot_ci(a,b,fn,nb=1000):
    out=[]
    for _ in range(nb):
        s=rng.choice(ucs,len(ucs),replace=True); ii=np.concatenate([idx[c] for c in s]); out.append(fn(a[ii],b[ii]))
    return np.percentile(out,[2.5,97.5]).tolist()
for name,(a,b) in {'selected(A4,B5)':(C.cl_A.values,C.cl_B.values),'k4':(C.A_k4.values,C.B_k4.values),'k5':(C.A_k5.values,C.B_k5.values),'k6':(C.A_k6.values,C.B_k6.values)}.items():
    r=agree(a,b); r['ARI_ci']=boot_ci(a,b,adjusted_rand_score); r['AMI_ci']=boot_ci(a,b,adjusted_mutual_info_score,300)
    perm=[adjusted_rand_score(a,rng.permutation(b)) for _ in range(500)]; r['perm_p']=(1+sum(x>=r['ARI'] for x in perm))/501
    # within-rep stability reference (subsample ARI from stability table)
    AG[name]=r; print(name,{k:(np.round(v,3) if not isinstance(v,list) else np.round(v,3).tolist()) for k,v in r.items()})
res['agreement']=AG
# 2022-only agreement
S=C[C.year==2022]; res['agreement_2022_selected']=agree(S.cl_A.values,S.cl_B.values)
# contingency
ct=pd.crosstab(C.cl_A,C.cl_B); ct.to_csv('results/contingency_selected.csv'); print(ct)
# normalized: share of B clusters' mass in best-matching A cluster (purity of B w.r.t. A)
res['purity_B_given_A']=float(ct.max(axis=1).sum()/ct.values.sum()); res['purity_A_given_B']=float(ct.max(axis=0).sum()/ct.values.sum())
# cluster profiles (raw indicators)
prof=[]
for r in 'AB':
    g=C.groupby(f'cl_{r}').agg(n=('country','size'),countries=('country','nunique'),gdp_pc=('gdp_pc','median'),energy_pc_kwh=('total_energy_per_capita_kwh','median'),fossil_sh=('fossil_sh','median'),carbon_int=('carbon_int','median'),elec_div=('elec_div','median'),netimp=('netimp_sh','median'),renew_sh=('renew_sh','median'))
    g['rep']=r; prof.append(g.reset_index().rename(columns={f'cl_{r}':'cluster'}))
pd.concat(prof).to_csv('results/cluster_profiles.csv',index=False); print(pd.concat(prof).round(1).to_string())
# eta^2 of strategic indicators by cluster label
def eta2(y,l):
    m=~np.isnan(y); y=y[m]; l=l[m]; g=pd.Series(y).groupby(l).transform('mean'); return float(((g-y.mean())**2).sum()/((y-y.mean())**2).sum())
e={}
for r in 'AB':
    e[r]={v:eta2(C[v].values.astype(float),C[f'cl_{r}'].values) for v in ['carbon_int','elec_div','import_exp','energy_int','renew_sh','fossil_sh','ln_gdp_pc','ln_energy_pc']}
res['eta2']=e; print(pd.DataFrame(e).round(3))
json.dump(res,open('results/p4_results.json','w'),indent=1,default=float)
