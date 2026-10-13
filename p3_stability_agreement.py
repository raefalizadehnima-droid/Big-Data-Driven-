import json,sys; sys.path.insert(0,'code')
from common import *
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
C=pd.read_pickle('results/common_sample.pkl')
countries=C.country.unique(); rng=np.random.default_rng(SEED)
NB=60
# subsample indices (shared by both reps => paired)
subs=[rng.choice(countries,int(0.8*len(countries)),replace=False) for _ in range(NB)]
M={r:build(r,C) for r in 'AB'}
rows=[]
for rep in 'AB':
    for k in range(2,11):
        ref=km(M[rep]['sc'],k).labels_
        aris=[]
        for s in subs:
            mask=C.country.isin(s).values; Cs=C[mask]
            m=build(rep,Cs); lab=km(m['sc'],k).labels_
            aris.append(adjusted_rand_score(ref[mask],lab))
        rows.append(dict(rep=rep,k=k,stab_mean=np.mean(aris),stab_sd=np.std(aris),stab_lo=np.percentile(aris,2.5),stab_hi=np.percentile(aris,97.5)))
St=pd.DataFrame(rows); St.to_csv('results/cluster_stability.csv',index=False); print(St.round(3).to_string())
# alternative algorithms
alt=[]
for rep in 'AB':
    sc=M[rep]['sc']
    for k in [3,4,5,6]:
        l_km=km(sc,k).labels_; l_w=AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(sc)
        l_g=GaussianMixture(k,n_init=5,random_state=SEED).fit_predict(sc)
        alt.append(dict(rep=rep,k=k,sil_km=silhouette_score(sc,l_km),sil_ward=silhouette_score(sc,l_w),sil_gmm=silhouette_score(sc,l_g),ARI_km_ward=adjusted_rand_score(l_km,l_w),ARI_km_gmm=adjusted_rand_score(l_km,l_g)))
Alt=pd.DataFrame(alt); Alt.to_csv('results/cluster_alt_algorithms.csv',index=False); print(Alt.round(3).to_string())
