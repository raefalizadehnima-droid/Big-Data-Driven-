import json,sys; sys.path.insert(0,'code')
from common import *
from scipy.stats import spearmanr, wilcoxon
from sklearn.impute import KNNImputer
rng=np.random.default_rng(SEED+1)
C=pd.read_pickle('results/common_labeled.pkl').reset_index(drop=True)
out={}
# ================= TEMPORAL =================
def trans(df,col,pairs=((2000,2005),(2005,2010),(2010,2015),(2015,2020))):
    P=df.pivot(index='country',columns='year',values=col); rows=[]
    for a,b in pairs:
        t=P[[a,b]].dropna(); t.columns=['from','to']; t['pair']=f'{a}-{b}'; rows.append(t)
    return pd.concat(rows)
T={}
for rep,k in [('A',4),('B',5),('A5',5),('B4',4),('A6',6),('B6',6)]:
    col={'A':'cl_A','B':'cl_B','A5':'A_k5','B4':'B_k4','A6':'A_k6','B6':'B_k6'}[rep]
    t=trans(C,col); pers=(t['from'].values==t['to'].values).mean()
    ari=adjusted_rand_score(t['from'],t['to'])
    cs=t.index.values; u=np.unique(cs); ix={c:np.where(cs==c)[0] for c in u}
    bs=[];ba=[]
    for _ in range(1000):
        s=rng.choice(u,len(u),replace=True); ii=np.concatenate([ix[c] for c in s]); tt=t.iloc[ii]
        bs.append((tt['from'].values==tt['to'].values).mean()); ba.append(adjusted_rand_score(tt['from'].values,tt['to'].values))
    M=pd.crosstab(t['from'].values,t['to'].values,normalize='index')
    # whole-period 2000->2022
    P=C.pivot(index='country',columns='year',values=col)[[2000,2022]].dropna()
    T[rep]=dict(n_transitions=len(t),persistence=pers,persistence_ci=np.percentile(bs,[2.5,97.5]).tolist(),ari_5yr=ari,ari_5yr_ci=np.percentile(ba,[2.5,97.5]).tolist(),n_2000_2022=len(P),changed_2000_2022=float((P[2000]!=P[2022]).mean()),ari_2000_2022=adjusted_rand_score(P[2000],P[2022]),
        trans_matrix=M.round(3).values.tolist(),k=k)
    if rep in('A','B'):
        M.to_csv(f'results/transition_matrix_{rep}.csv')
        pd.crosstab(P[2000],P[2022]).to_csv(f'results/flow_2000_2022_{rep}.csv')
out['temporal']=T; print({r:{k:v for k,v in T[r].items() if k!='trans_matrix'} for r in T})
# net flow into the lowest-carbon cluster
prof=pd.read_csv('results/cluster_profiles.csv')
fl={}
for rep in 'AB':
    pr=prof[prof.rep==rep]; low=int(pr.loc[pr.carbon_int.idxmin(),'cluster'])
    P=C.pivot(index='country',columns='year',values=f'cl_{rep}')[[2000,2022]].dropna()
    fl[rep]=dict(low_carbon_cluster=low,n_2000=int((P[2000]==low).sum()),n_2022=int((P[2022]==low).sum()),entered=int(((P[2000]!=low)&(P[2022]==low)).sum()),left=int(((P[2000]==low)&(P[2022]!=low)).sum()),n=len(P))
out['lowcarbon_flow']=fl; print(fl)
# dispersion (RMS distance to pooled centroid / sqrt(p)) by year, bootstrap over countries
disp={}
for rep in 'AB':
    m=build(rep,C); Z=m['Z'].values; cen=Z.mean(0); dist=np.sqrt(((Z-cen)**2).sum(1)/Z.shape[1]); C[f'disp_{rep}']=dist
    cc=C.groupby(['year'])[f'disp_{rep}'].mean(); 
    ucs=C.country.unique(); bsd=[]
    for _ in range(500):
        s=rng.choice(ucs,len(ucs),replace=True); g=C.set_index('country').loc[s].reset_index(); bsd.append(g[g.year==2022][f'disp_{rep}'].mean()-g[g.year==2000][f'disp_{rep}'].mean())
    disp[rep]=dict(by_year=cc.round(3).to_dict(),delta_2000_2022=float(cc[2022]-cc[2000]),ci=np.percentile(bsd,[2.5,97.5]).tolist())
out['dispersion']=disp; print(disp)
# centroid trajectories in PC1-PC2
traj={}
for rep in 'AB':
    m=build(rep,C); sc=m['sc']; df=pd.DataFrame({'year':C.year,'pc1':sc[:,0],'pc2':sc[:,1]}); traj[rep]=df.groupby('year').mean().round(3).to_dict()
out['centroid_traj']=traj
C.to_pickle('results/common_labeled_disp.pkl')
# ================= ROBUSTNESS =================
countries=C.country.unique(); NS=100
subs=[rng.choice(countries,int(0.8*len(countries)),replace=False) for _ in range(NS)]
K={'A':4,'B':5}; ref={r:C[f'cl_{r}'].values for r in 'AB'}
stab={r:[] for r in 'AB'}
for s in subs:
    mask=C.country.isin(s).values
    for r in 'AB':
        m=build(r,C[mask]); lab=km(m['sc'],K[r]).labels_; stab[r].append(adjusted_rand_score(ref[r][mask],lab))
sa,sb=np.array(stab['A']),np.array(stab['B'])
w=wilcoxon(sb,sa); diff=sb-sa
rbc=(np.sum(diff>0)-np.sum(diff<0))/np.sum(diff!=0)
bd=[np.mean(rng.choice(diff,len(diff))) for _ in range(2000)]
out['cluster_stability_paired']=dict(A_mean=sa.mean(),A_ci=np.percentile(sa,[2.5,97.5]).tolist(),B_mean=sb.mean(),B_ci=np.percentile(sb,[2.5,97.5]).tolist(),mean_diff_B_minus_A=diff.mean(),diff_ci=np.percentile(bd,[2.5,97.5]).tolist(),wilcoxon_stat=float(w.statistic),wilcoxon_p=float(w.pvalue),rank_biserial=float(rbc),n_draws=NS)
print(out['cluster_stability_paired'])
np.save('results/stab_A.npy',sa); np.save('results/stab_B.npy',sb)
# leave-one-variable-out
lov={}
for r in 'AB':
    feats=A_FEATS if r=='A' else B_FEATS; res=[]
    for f in feats:
        keep=[x for x in feats if x!=f]
        import common as cm
        old=(cm.A_FEATS[:],cm.B_FEATS[:])
        if r=='A': cm.A_FEATS[:]=keep
        else: cm.B_FEATS[:]=keep
        m=cm.build(r,C); lab=km(m['sc'],K[r]).labels_; res.append((f,adjusted_rand_score(ref[r],lab)))
        cm.A_FEATS[:],cm.B_FEATS[:]=old
    v=np.array([x[1] for x in res]); lov[r]=dict(mean=v.mean(),min=v.min(),max=v.max(),argmin=res[int(v.argmin())][0],all={a:round(b,3) for a,b in res})
out['leave_one_variable_out']=lov; print({r:{k:v for k,v in lov[r].items() if k!='all'} for r in lov}); print(lov['B']['all'])
# normalisation / outlier alternatives
alt={}
for r in 'AB':
    alt[r]={}
    for nm in ['rank','raw']:
        m=build(r,C,norm=nm); alt[r][nm]=adjusted_rand_score(ref[r],km(m['sc'],K[r]).labels_)
    # winsor 5-95
    feats=A_FEATS if r=='A' else B_FEATS; X=tf(C,feats,'A' if r=='A' else 'B'); Z=zscore(winsor(X,5,95)); ev,thr,npc=parallel_analysis(Z.values)
    sc=PCA(n_components=max(npc,2),random_state=SEED).fit_transform(Z.values); alt[r]['winsor5-95']=adjusted_rand_score(ref[r],km(sc,K[r]).labels_)
    # drop top 2.5% Mahalanobis-like (z-distance) outliers then refit and compare on retained rows
    m=build(r,C); dist=np.linalg.norm(m['Z'].values,axis=1); keepm=dist<=np.percentile(dist,97.5); m2=build(r,C[keepm]); alt[r]['drop_top2.5pct_outliers']=adjusted_rand_score(ref[r][keepm],km(m2['sc'],K[r]).labels_)
out['normalisation_outlier_alt']=alt; print(alt)
# missing-data treatment: kNN imputation on all ISO rows at snapshot years
S=snapshot_rows(); res={}
for r in 'AB':
    feats=A_FEATS if r=='A' else B_FEATS; X=tf(S,feats,'A' if r=='A' else 'B'); frac=X.notna().mean(axis=1); keepm=(frac>=0.75)
    Xs=winsor(X[keepm]); Zs=zscore(Xs); Zi=pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(Zs),index=Zs.index,columns=Zs.columns)
    ev,thr,npc=parallel_analysis(Zi.values); sc=PCA(n_components=max(npc,2),random_state=SEED).fit_transform(Zi.values); lab=pd.Series(km(sc,K[r]).labels_,index=Zi.index)
    comm=C.copy(); 
    # map common rows back via (country,year)
    key=S.loc[Zi.index,['country','year']].assign(lab=lab.values); mg=C[['country','year']].assign(ref=ref[r]).merge(key,on=['country','year'])
    res[r]=dict(n_rows_imputed_set=int(keepm.sum()),n_rows_with_any_imputation=int((X[keepm].isna().any(axis=1)).sum()),countries=int(S[keepm].country.nunique()),ARI_vs_baseline_on_common=adjusted_rand_score(mg.ref,mg.lab),n_common=len(mg))
out['knn_imputation']=res; print(res)
# temporal holdout: train <=2010, predict >=2015
hold={}
for r in 'AB':
    feats=A_FEATS if r=='A' else B_FEATS; X=tf(C,feats,'A' if r=='A' else 'B'); tr=(C.year<=2010).values
    a,b=np.nanpercentile(X[tr],1,axis=0),np.nanpercentile(X[tr],99,axis=0); Xw=X.clip(a,b,axis=1); mu,sd=Xw[tr].mean(),Xw[tr].std(ddof=0); Z=(Xw-mu)/sd
    ev,thr,npc=parallel_analysis(Z[tr].values); pca=PCA(n_components=max(npc,2),random_state=SEED).fit(Z[tr].values)
    kmod=km(pca.transform(Z[tr].values),K[r]); pred=kmod.predict(pca.transform(Z[~tr].values))
    hold[r]=dict(n_train=int(tr.sum()),n_test=int((~tr).sum()),ARI_vs_pooled_labels=adjusted_rand_score(ref[r][~tr],pred),sil_test=float(silhouette_score(pca.transform(Z[~tr].values),pred)))
out['temporal_holdout']=hold; print(hold)
# ranking stability under subsampling (paired)

R=pd.read_pickle('results/snapshot2022_ranked.pkl').reset_index(drop=True)
OBJ={'A':[('fossil_sh',-1),('energy_int',-1),('total_energy_per_capita_kwh',1)],'B':[('carbon_int',-1),('elec_div',1),('import_exp',-1),('total_energy_per_capita_kwh',1)]}
def al(df,rep): return np.column_stack([df[c].values*s_ for c,s_ in OBJ[rep]]).astype(float)
def nm(X):
    a,b=np.nanpercentile(X,2.5,axis=0),np.nanpercentile(X,97.5,axis=0); X=np.clip(X,a,b); return (X-X.min(0))/(X.max(0)-X.min(0))
def rk(s_): return pd.Series(s_).rank(ascending=False,method='min').values
n=len(R); rst={'A':[],'B':[]}; rtop={'A':[],'B':[]}; NS2=300
base={'A':R.rankA.values,'B':R.rankB.values}
for _ in range(NS2):
    ii=np.sort(rng.choice(n,int(0.8*n),replace=False))
    for r,w in (('A',np.ones(3)/3),('B',np.ones(4)/4)):
        rr=rk(nm(al(R.iloc[ii],r))@w); rst[r].append(spearmanr(rr,base[r][ii])[0]); rtop[r].append(len(set(np.argsort(rr)[:10])&set(np.argsort(base[r][ii])[:10])))
ra,rb=np.array(rst['A']),np.array(rst['B']); dd=rb-ra; ww=wilcoxon(rb,ra)
out['rank_stability_subsample']=dict(A_mean=ra.mean(),A_ci=np.percentile(ra,[2.5,97.5]).tolist(),B_mean=rb.mean(),B_ci=np.percentile(rb,[2.5,97.5]).tolist(),diff_B_minus_A=dd.mean(),wilcoxon_p=float(ww.pvalue),rank_biserial=float((np.sum(dd>0)-np.sum(dd<0))/np.sum(dd!=0)),top10_retained_A=float(np.mean(rtop['A'])),top10_retained_B=float(np.mean(rtop['B'])),n_draws=NS2)
print(out['rank_stability_subsample'])
json.dump(out,open('results/p6_results.json','w'),indent=1,default=float)
