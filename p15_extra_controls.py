"""Extra controls: (1) feature screening using training years only, (2) compositional (CLR) transformation, (3) population-weighted clustering and references."""
import json,sys; sys.path.insert(0,'code')
from common import *
import common as cm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
out={}
d=pd.read_pickle('results/panel.pkl')
POOL=['ln_gdp_pc','ln_energy_pc','ln_elecdem_pc','carbon_int','coal_sh','gas_sh','oil_sh','nuclear_sh','hydro_sh','wind_sh','solar_sh','bio_sh','netimp_sh','ln1p_coalprod_pc','ln1p_oilprod_pc','ln1p_gasprod_pc']
# (1) screening on 2000-2010 only
def screen(df):
    cov=df[POOL].notna().mean()*100; R=df[POOL].corr(method='spearman'); D=1-R.abs().values; np.fill_diagonal(D,0)
    cl=fcluster(linkage(squareform(D,checks=False),'average'),t=0.20,criterion='distance'); sel=[]
    for g in np.unique(cl):
        mem=[POOL[i] for i in np.where(cl==g)[0]]; best=max(mem,key=lambda v:cov[v]); top=[v for v in mem if cov[best]-cov[v]<=0.5]
        pick=[v for v in top if v in A_FEATS]; sel.append(pick[0] if pick else sorted(top)[0])
    return sorted(sel),cov
sel_all,_=screen(d); sel_tr,cov_tr=screen(d[d.year<=2010])
out['screen_all_years']=sel_all; out['screen_training_years']=sel_tr; out['screen_identical']=bool(sel_all==sel_tr); out['screen_coverage_min_train']=float(cov_tr.min())
out['B_feats_sorted']=sorted(B_FEATS); out['B_matches_all_years']=bool(sorted(B_FEATS)==sel_all)
# (2) CLR
C=pd.read_pickle('results/common_labeled.pkl').reset_index(drop=True)
src=['coal_sh','gas_sh','oil_sh','nuclear_sh','hydro_sh','wind_sh','solar_sh','bio_sh','othren_sh']
ok=C[src].notna().all(axis=1); Cc=C[ok].reset_index(drop=True); out['clr_rows']=int(ok.sum()); out['clr_rows_total']=len(C)
def clr_feats(df,pc):
    X=df[src].values/100.0+pc/100.0; X=X/X.sum(1,keepdims=True); L=np.log(X); L=L-L.mean(1,keepdims=True)
    return pd.DataFrame(L,columns=[s+'_clr' for s in src],index=df.index)
base=[v for v in B_FEATS if not v.endswith('_sh') or v=='netimp_sh']   # non-share features incl. net imports
keep_src=['coal_sh','gas_sh','oil_sh','nuclear_sh','wind_sh','solar_sh','bio_sh']
res={}
refB=Cc.cl_B.values; refA=Cc.cl_A.values
for pc in (0.01,0.1,1.0):
    X=tf(Cc,base,'B'); X=pd.concat([X,clr_feats(Cc,pc)[[s+'_clr' for s in keep_src]]],axis=1)
    Z=zscore(winsor(X)); ev,thr,npc=parallel_analysis(Z.values); pca=PCA(n_components=max(npc,2),random_state=SEED).fit(Z.values); sc=pca.transform(Z.values)
    lab=km(sc,5).labels_
    res[str(pc)]=dict(npc=int(npc),ARI_vs_sqrt_B=adjusted_rand_score(refB,lab),ARI_vs_A4=adjusted_rand_score(refA,lab),sil=float(silhouette_score(sc,lab)))
out['clr']=res; out['ARI_A4_vs_B_sqrt_same_rows']=adjusted_rand_score(refA,refB)
# (3) population weights
pop=C.population.values.astype(float); w=pop/pop.mean(); lab_w={}; w_res={}
for rep,k,col in (('A',4,'cl_A'),('B',5,'cl_B')):
    m=build(rep,C); sc=m['sc']; kw=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(sc,sample_weight=w); lab_w[rep]=kw.labels_
    # best-match agreement (population-weighted share of rows keeping their cluster after optimal relabelling)
    ct=pd.crosstab(C[col].values,kw.labels_).reindex(index=range(k),columns=range(k),fill_value=0).values
    ri,ci=linear_sum_assignment(-ct); mp=dict(zip(ci,ri)); same=np.array([mp[l]==o for l,o in zip(kw.labels_,C[col].values)])
    w_res[rep]=dict(ARI_weighted_vs_unweighted=adjusted_rand_score(C[col].values,kw.labels_),share_same_cluster_rows=float(same.mean()),share_same_cluster_population=float((same*pop).sum()/pop.sum()))
w_res['ARI_A_vs_B_weighted']=adjusted_rand_score(lab_w['A'],lab_w['B']); w_res['ARI_A_vs_B_unweighted']=adjusted_rand_score(C.cl_A,C.cl_B)
# peer reference agreement weighted by 2022 population (same fixed-objective score as p13)
need=['fossil_sh','energy_int','total_energy_per_capita_kwh','carbon_int','elec_div','import_exp']
idx=np.where((C.year==2022).values & C[need].notna().all(axis=1).values)[0]; S=C.iloc[idx].reset_index(drop=True); n=len(S)
OBJ=[('carbon_int',-1),('elec_div',1),('import_exp',-1),('total_energy_per_capita_kwh',1),('energy_int',-1)]
F=np.column_stack([S[c].values*s for c,s in OBJ]).astype(float); cl=np.clip(F,np.nanpercentile(F,2.5,axis=0),np.nanpercentile(F,97.5,axis=0)); score=((cl-cl.min(0))/(cl.max(0)-cl.min(0))).mean(1)
def refs(lab):
    r=np.empty(n,int)
    for c in np.unique(lab):
        m=np.where(lab==c)[0]; r[m]=m[np.argmax(score[m])]
    return r
rA,rB=refs(S.cl_A.values),refs(S.cl_B.values); p22=S.population.values.astype(float)
w_res['same_reference_unweighted']=float((rA==rB).mean()); w_res['same_reference_population_weighted']=float(((rA==rB)*p22).sum()/p22.sum())
wl=pd.Series(lab_w['A'],index=C.index).iloc[idx].values; wlb=pd.Series(lab_w['B'],index=C.index).iloc[idx].values
rAw,rBw=refs(wl),refs(wlb); w_res['same_reference_with_weighted_clusters']=float((rAw==rBw).mean()); w_res['n_2022']=n
out['pop_weighted']=w_res
json.dump(out,open('results/p15_extra_controls.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
