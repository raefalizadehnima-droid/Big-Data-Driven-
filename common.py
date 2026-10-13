import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score
SEED=20260411
YEARS=[2000,2005,2010,2015,2020,2022]
d=pd.read_pickle('results/panel.pkl')
d=d.reset_index(drop=True)
# ---- feature definitions ----
A_FEATS=['ln_gdp_pc','ln_energy_pc','fossil_sh']
# B selection: pool clusters (|rho|>0.80, average linkage) -> representative = max coverage; ties (<0.5pp) -> variable used in A, else alphabetical
B_FEATS=['carbon_int','oil_sh','ln_energy_pc','ln1p_oilprod_pc','gas_sh','wind_sh','bio_sh','solar_sh','coal_sh','ln1p_coalprod_pc','nuclear_sh','netimp_sh']
def tf(df,cols,mode):
    X=df[cols].copy()
    if mode=='B':
        for c in cols:
            if c.endswith('_sh'): X[c]=np.sqrt(X[c].clip(0,100)/100)
        if 'netimp_sh' in cols: X['netimp_sh']=np.sign(df.netimp_sh)*np.sqrt(df.netimp_sh.abs().clip(0,100)/100)
    return X
def winsor(X,lo=1,hi=99,ref=None):
    ref=X if ref is None else ref
    a,b=np.nanpercentile(ref,lo,axis=0),np.nanpercentile(ref,hi,axis=0)
    return X.clip(a,b,axis=1)
def zscore(X,ref=None):
    ref=X if ref is None else ref
    return (X-ref.mean())/ref.std(ddof=0)
def parallel_analysis(Z,nperm=200,seed=SEED):
    rng=np.random.default_rng(seed); Z=np.asarray(Z)
    ev=np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]
    sims=np.empty((nperm,Z.shape[1]))
    for i in range(nperm):
        Zp=np.column_stack([rng.permutation(Z[:,j]) for j in range(Z.shape[1])])
        sims[i]=np.sort(np.linalg.eigvalsh(np.corrcoef(Zp.T)))[::-1]
    thr=np.percentile(sims,95,axis=0)
    return ev,thr,max(2,int((ev>thr).sum()))
def build(rep,rows,pw=True,norm='z'):
    """rows: DataFrame of country-years (complete). returns Z (standardised features), scores (PCs), npc, pca"""
    feats=A_FEATS if rep=='A' else B_FEATS
    X=tf(rows,feats,'A' if rep=='A' else 'B')
    if norm=='raw': Xw=X
    else: Xw=winsor(X)
    if norm=='rank': Z=Xw.rank(pct=True); Z=(Z-Z.mean())/Z.std(ddof=0)
    else: Z=zscore(Xw)
    ev,thr,npc=parallel_analysis(Z.values)
    pca=PCA(n_components=min(len(feats),max(npc,2)),random_state=SEED).fit(Z.values)
    sc=pca.transform(Z.values)
    return dict(Z=Z,sc=sc,npc=npc,pca=pca,ev=ev,thr=thr,feats=feats)
def km(sc,k,seed=SEED):
    return KMeans(n_clusters=k,n_init=20,random_state=seed).fit(sc)
def vi(a,b):
    from sklearn.metrics import mutual_info_score
    a=np.asarray(a);b=np.asarray(b)
    def H(x):
        _,c=np.unique(x,return_counts=True);p=c/c.sum();return -(p*np.log(p)).sum()
    return H(a)+H(b)-2*mutual_info_score(a,b)
def snapshot_rows(years=YEARS):
    return d[d.year.isin(years)].copy()
