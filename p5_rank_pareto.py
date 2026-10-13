import json,sys; sys.path.insert(0,'code')
from common import *
from scipy.stats import spearmanr, kendalltau
rng=np.random.default_rng(SEED)
C=pd.read_pickle('results/common_labeled.pkl'); S=C[C.year==2022].reset_index(drop=True); n_before=len(S); S=S.dropna(subset=['fossil_sh','energy_int','total_energy_per_capita_kwh','carbon_int','elec_div','import_exp']).reset_index(drop=True)
# objectives: (column, direction) ; direction +1 maximise, -1 minimise
OBJ={'A':[('fossil_sh',-1),('energy_int',-1),('total_energy_per_capita_kwh',1)],
     'B':[('carbon_int',-1),('elec_div',1),('import_exp',-1),('total_energy_per_capita_kwh',1)]}
OBJ['B5']=OBJ['B']+[('energy_int',-1)]
def aligned(df,rep): return np.column_stack([df[c].values*s for c,s in OBJ[rep]]).astype(float)
def norm(X,mode='minmax',lo=2.5,hi=97.5):
    X=X.copy()
    if mode=='rank': return (pd.DataFrame(X).rank(pct=True).values)
    if mode!='raw':
        a,b=np.nanpercentile(X,lo,axis=0),np.nanpercentile(X,hi,axis=0); X=np.clip(X,a,b)
    if mode=='z': return (X-X.mean(0))/X.std(0)
    return (X-X.min(0))/(X.max(0)-X.min(0))
def score(N,w): return N@w
def ranks(s): return pd.Series(s).rank(ascending=False,method='min').values
def nds(X):
    """non-dominated sorting for maximisation; returns front index (0=Pareto-efficient)"""
    n=len(X); front=np.full(n,-1); rem=np.arange(n); f=0
    while len(rem):
        sub=X[rem]; nd=np.ones(len(rem),bool)
        for i in range(len(rem)):
            dom=(sub>=sub[i]).all(1)&(sub>sub[i]).any(1)
            if dom.any(): nd[i]=False
        front[rem[nd]]=f; rem=rem[~nd]; f+=1
    return front
out={}
XA,XB,XB5=aligned(S,'A'),aligned(S,'B'),aligned(S,'B5')
n=len(S); out['n_countries_2022_common']=n; out['n_2022_common_before_objective_filter']=n_before
wA,wB=np.ones(3)/3,np.ones(4)/4
NA,NB=norm(XA),norm(XB)
sA,sB=score(NA,wA),score(NB,wB); rA,rB=ranks(sA),ranks(sB)
S['rankA'],S['rankB'],S['scoreA'],S['scoreB']=rA,rB,sA,sB
# ---- rank comparison ----
def cmp(rA,rB):
    d=np.abs(rA-rB); n=len(rA)
    q=lambda r:np.minimum(((r-1)*5//n).astype(int),4)
    return dict(spearman=spearmanr(rA,rB)[0],kendall=kendalltau(rA,rB)[0],pair_reversal=(1-kendalltau(rA,rB)[0])/2,mean_abs_disp=d.mean(),median_abs_disp=float(np.median(d)),p90_disp=float(np.percentile(d,90)),max_disp=int(d.max()),share_disp_ge10pct=float((d>=0.1*n).mean()),share_disp_ge25pct=float((d>=0.25*n).mean()),
        top10_overlap=len(set(np.argsort(rA)[:10])&set(np.argsort(rB)[:10])),top20_overlap=len(set(np.argsort(rA)[:20])&set(np.argsort(rB)[:20])),quintile_change_share=float((q(rA)!=q(rB)).mean()),quintile_change_ge2=float((np.abs(q(rA)-q(rB))>=2).mean()))
main=cmp(rA,rB); out['rank_main']=main
def bci(fn,nb=2000):
    v=[]
    for _ in range(nb):
        i=rng.choice(n,n,replace=True); v.append(fn(i))
    return np.percentile(v,[2.5,97.5]).tolist()
out['rank_main_ci']=dict(spearman=bci(lambda i:spearmanr(rA[i],rB[i])[0]),kendall=bci(lambda i:kendalltau(rA[i],rB[i])[0],500))
print(json.dumps(out,indent=1,default=float))
# variants
NB5=norm(XB5); sB5=score(NB5,np.ones(5)/5); rB5=ranks(sB5); out['rank_A_vs_B5']=cmp(rA,rB5)
# normalisation / aggregation sensitivity (cross-rep and within-rep vs baseline)
sens=[]
for mode in ['minmax','z','rank','raw']:
    a=ranks(score(norm(XA,mode),wA)); b=ranks(score(norm(XB,mode),wB))
    sens.append(dict(norm=mode,cross_tau=kendalltau(a,b)[0],cross_rho=spearmanr(a,b)[0],A_vs_base_rho=spearmanr(a,rA)[0],B_vs_base_rho=spearmanr(b,rB)[0],top10_overlap=len(set(np.argsort(a)[:10])&set(np.argsort(b)[:10]))))
for lo,hi in [(0,100),(1,99),(5,95)]:
    a=ranks(score(norm(XA,'minmax',lo,hi),wA)); b=ranks(score(norm(XB,'minmax',lo,hi),wB))
    sens.append(dict(norm=f'minmax_winsor{lo}-{hi}',cross_tau=kendalltau(a,b)[0],cross_rho=spearmanr(a,b)[0],A_vs_base_rho=spearmanr(a,rA)[0],B_vs_base_rho=spearmanr(b,rB)[0],top10_overlap=len(set(np.argsort(a)[:10])&set(np.argsort(b)[:10]))))
# Copeland (pairwise dominance) aggregation
def copeland(X):
    N=len(X); sc=np.zeros(N)
    for j in range(N): sc[j]=((X[j]>X).sum(1)-(X[j]<X).sum(1)).clip(-X.shape[1],X.shape[1]).sum()
    return sc
a=ranks(copeland(XA)); b=ranks(copeland(XB))
sens.append(dict(norm='copeland_pairwise',cross_tau=kendalltau(a,b)[0],cross_rho=spearmanr(a,b)[0],A_vs_base_rho=spearmanr(a,rA)[0],B_vs_base_rho=spearmanr(b,rB)[0],top10_overlap=len(set(np.argsort(a)[:10])&set(np.argsort(b)[:10]))))
pd.DataFrame(sens).to_csv('results/rank_norm_sensitivity.csv',index=False); print(pd.DataFrame(sens).round(3).to_string())
# ---- weight perturbation: Dirichlet(1) ----
NW=2000; WA=rng.dirichlet(np.ones(3),NW); WB=rng.dirichlet(np.ones(4),NW)
tau_cross=[];tauA=[];tauB=[];RA=[];RB=[]
for i in range(NW):
    a=ranks(score(NA,WA[i])); b=ranks(score(NB,WB[i])); RA.append(a);RB.append(b)
    tau_cross.append(kendalltau(a,b)[0]); tauA.append(kendalltau(a,rA)[0]); tauB.append(kendalltau(b,rB)[0])
RA,RB=np.array(RA),np.array(RB)
out['weights']=dict(cross_tau_mean=float(np.mean(tau_cross)),cross_tau_ci=np.percentile(tau_cross,[2.5,97.5]).tolist(),A_within_tau_mean=float(np.mean(tauA)),A_within_tau_ci=np.percentile(tauA,[2.5,97.5]).tolist(),B_within_tau_mean=float(np.mean(tauB)),B_within_tau_ci=np.percentile(tauB,[2.5,97.5]).tolist(),
    share_withinA_ge_crossbase=float(np.mean(np.array(tauA)>=main['kendall'])),share_withinB_ge_crossbase=float(np.mean(np.array(tauB)>=main['kendall'])),
    A_rank_interval_width_mean=float(np.mean(np.percentile(RA,95,axis=0)-np.percentile(RA,5,axis=0))),B_rank_interval_width_mean=float(np.mean(np.percentile(RB,95,axis=0)-np.percentile(RB,5,axis=0))),
    A_top10_stable=int((((RA<=10).mean(0))>=0.9).sum()),B_top10_stable=int((((RB<=10).mean(0))>=0.9).sum()),
    A_top20_freq_ge50=int((((RA<=20).mean(0))>=0.5).sum()),B_top20_freq_ge50=int((((RB<=20).mean(0))>=0.5).sum()))
np.save('results/tau_cross_weights.npy',np.array(tau_cross)); np.save('results/tau_withinA.npy',np.array(tauA)); np.save('results/tau_withinB.npy',np.array(tauB))
print(out['weights'])
# ---- Pareto ----
fA,fB,fB5=nds(XA),nds(XB),nds(XB5)
S['frontA'],S['frontB'],S['frontB5']=fA,fB,fB5
EA,EB,EB5=set(S.country[fA==0]),set(S.country[fB==0]),set(S.country[fB5==0])
par=dict(nA=len(EA),nB=len(EB),nB5=len(EB5),both=len(EA&EB),onlyA=len(EA-EB),onlyB=len(EB-EA),jaccard=len(EA&EB)/len(EA|EB),both_B5=len(EA&EB5),jaccard_B5=len(EA&EB5)/len(EA|EB5),n_fronts_A=int(fA.max()+1),n_fronts_B=int(fB.max()+1))
# null: independence of objectives (column-permutation) expected efficient count
nullA=[(nds(np.column_stack([rng.permutation(XA[:,j]) for j in range(3)]))==0).sum() for _ in range(300)]
nullB=[(nds(np.column_stack([rng.permutation(XB[:,j]) for j in range(4)]))==0).sum() for _ in range(300)]
par['null_A_mean']=float(np.mean(nullA)); par['null_A_ci']=np.percentile(nullA,[2.5,97.5]).tolist(); par['null_B_mean']=float(np.mean(nullB)); par['null_B_ci']=np.percentile(nullB,[2.5,97.5]).tolist()
# B restricted to 3-objective subsets
sub=[]
import itertools
for comb in itertools.combinations(range(4),3):
    f=nds(XB[:,list(comb)]); sub.append(dict(objs=[OBJ['B'][i][0] for i in comb],n_eff=int((f==0).sum()),overlap_with_A=len(set(S.country[f==0])&EA)))
par['B_3obj_subsets']=sub
# objective correlations (aligned)
par['obj_corr_A']=np.corrcoef(XA.T).round(2).tolist(); par['obj_corr_B']=np.corrcoef(XB.T).round(2).tolist()
# perturbation of Pareto membership (noise sd = 5% of min-max range), 500 draws
def freq_eff(X,nd=500,sd=0.05):
    N=norm(X,'raw'); N=(X-X.min(0))/(X.max(0)-X.min(0)); f=np.zeros(len(X))
    for _ in range(nd): f+=(nds(N+rng.normal(0,sd,N.shape))==0)
    return f/nd
frA,frB=freq_eff(XA),freq_eff(XB)
S['pfreqA'],S['pfreqB']=frA,frB
par['robust_eff_A_ge90']=int((frA>=0.9).sum()); par['robust_eff_B_ge90']=int((frB>=0.9).sum())
par['robust_both_ge90']=len(set(S.country[frA>=0.9])&set(S.country[frB>=0.9]))
par['robust_eff_A_among_efficient']=float((frA[fA==0]>=0.9).mean()); par['robust_eff_B_among_efficient']=float((frB[fB==0]>=0.9).mean())
par['mean_freq_eff_members_A']=float(frA[fA==0].mean()); par['mean_freq_eff_members_B']=float(frB[fB==0].mean())
out['pareto']=par; print(json.dumps({k:v for k,v in par.items() if 'corr' not in k},indent=1,default=float))
S.to_pickle('results/snapshot2022_ranked.pkl')
# ---- coverage consequence: B evaluable set vs A ----
full=d_all=pd.read_pickle('results/panel.pkl'); F=full[full.year==2022].copy()
Bcols=[c for c,_ in OBJ['B']]; Acols=[c for c,_ in OBJ['A']]
okB=F[Bcols].notna().all(axis=1); okA=F[Acols].notna().all(axis=1)
Fb=F[okB].reset_index(drop=True); XFb=aligned(Fb,'B'); NFb=norm(XFb); Fb['scoreB']=NFb@wB; Fb['rankB_full']=ranks(Fb.scoreB.values); Fb['front_full']=nds(XFb); Fb['evaluable_A']=Fb.country.isin(F[okA].country)
cov=dict(n_B_eval=int(okB.sum()),n_A_eval=int(okA.sum()),n_B_only=int((okB&~okA).sum()),n_A_only=int((okA&~okB).sum()),
  top20_B_full_not_evaluable_A=int((~Fb[Fb.rankB_full<=20].evaluable_A).sum()),eff_B_full=int((Fb.front_full==0).sum()),eff_B_full_not_evaluable_A=int(((Fb.front_full==0)&~Fb.evaluable_A).sum()),
  B_only_countries_pop_M=float(F[okB&~okA].population.sum()/1e6),A_eval_pop_M=float(F[okA].population.sum()/1e6),total_pop_M_B=float(F[okB].population.sum()/1e6))
out['coverage_consequence']=cov; print(cov)
Fb.to_pickle('results/snapshot2022_Bfull.pkl')
json.dump(out,open('results/p5_results.json','w'),indent=1,default=float)
