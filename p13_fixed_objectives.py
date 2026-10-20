"""Control: one fixed objective set for both representations; representation-dependent decision rules only."""
import json,sys; sys.path.insert(0,'code')
from common import *
from scipy.stats import kendalltau
rng=np.random.default_rng(SEED+13)
C=pd.read_pickle('results/common_labeled.pkl').reset_index(drop=True)
R2=pd.read_pickle('results/snapshot2022_ranked.pkl')
need=['fossil_sh','energy_int','total_energy_per_capita_kwh','carbon_int','elec_div','import_exp']
y22=np.where((C.year==2022).values & C[need].notna().all(axis=1).values)[0]; S=C.iloc[y22].reset_index(drop=True); n=len(S)
assert n==len(R2) and set(S.country)==set(R2.country)
OBJ=[('carbon_int',-1),('elec_div',1),('import_exp',-1),('total_energy_per_capita_kwh',1),('energy_int',-1)]
F=np.column_stack([S[c].values*s for c,s in OBJ]).astype(float)
cl=np.clip(F,np.nanpercentile(F,2.5,axis=0),np.nanpercentile(F,97.5,axis=0)); Fn=(cl-cl.min(0))/(cl.max(0)-cl.min(0)); score=Fn.mean(1)
rank_fixed=pd.Series(score).rank(ascending=False,method='min').values
def nd(X): return np.array([not ((X>=X[i]).all(1)&(X>X[i]).any(1)).any() for i in range(len(X))])
out={'n':n,'objectives':[c for c,_ in OBJ]}
# (1) global ranking and global Pareto set depend only on the objective values: identical for every representation (by construction)
par_fixed=nd(F); out['global_pareto_fixed']=int(par_fixed.sum()); out['ranking_identical_across_representations']=True
# (2) effect of the objective set itself (earlier A and B rankings against the fixed set)
RA=R2.set_index('country').loc[S.country]; 
def jac(a,b): return float((a&b).sum()/max(1,(a|b).sum()))
out['objective_set_effect']=dict(tau_A_obj_vs_fixed=float(kendalltau(RA.rankA.values,rank_fixed)[0]),tau_B_obj_vs_fixed=float(kendalltau(RA.rankB.values,rank_fixed)[0]),tau_A_obj_vs_B_obj=float(kendalltau(RA.rankA.values,RA.rankB.values)[0]),
   pareto_A=int((RA.frontA.values==0).sum()),pareto_B=int((RA.frontB.values==0).sum()),pareto_fixed=int(par_fixed.sum()),jac_A_vs_fixed=jac(RA.frontA.values==0,par_fixed),jac_B_vs_fixed=jac(RA.frontB.values==0,par_fixed),jac_A_vs_B=jac(RA.frontA.values==0,RA.frontB.values==0))
# (3) representation-dependent decision rules
Z={}
for rep in 'AB':
    m=build(rep,C); Z[rep]=m['Z'].values[y22]
leaders=np.argsort(-score)[:10]; nonlead=np.setdiff1d(np.arange(n),leaders)
def rules(lab,Zr):
    ref=np.empty(n,int); par=np.zeros(n,bool)
    for c in np.unique(lab):
        m=np.where(lab==c)[0]; best=m[np.argmax(score[m])]; ref[m]=best; par[m[nd(F[m])]]=True
    D=np.sqrt(((Zr[:,None,:]-Zr[None,leaders,:])**2).sum(2)).mean(1); feas=pd.Series(D[nonlead]).rank().values
    gap=score[ref]-score
    return ref,par,feas,gap
def cmpr(labA,labB):
    rA,pA,fA,gA=rules(labA,Z['A']); rB,pB,fB,gB=rules(labB,Z['B'])
    same=(rA==rB)
    return dict(same_reference=float(same.mean()),same_reference_excl_self=float((rA==rB)[(rA!=np.arange(n))&(rB!=np.arange(n))].mean()),jac_cluster_pareto=jac(pA,pB),size_A=int(pA.sum()),size_B=int(pB.sum()),
        tau_feasibility=float(kendalltau(fA,fB)[0]),top20_feasible_overlap=len(set(np.argsort(fA)[:20])&set(np.argsort(fB)[:20])),mean_gap_A=float(gA.mean()),mean_gap_B=float(gB.mean()),corr_gap=float(np.corrcoef(gA,gB)[0,1]),
        leaders_A=int((rA==np.arange(n)).sum()),leaders_B=int((rB==np.arange(n)).sum())),(rA,rB)
res={}
for name,(a,b) in {'A4_vs_B5':('cl_A','cl_B'),'k4':('A_k4','B_k4'),'k5':('A_k5','B_k5'),'k6':('A_k6','B_k6')}.items():
    r,_=cmpr(S[a].values,S[b].values); 
    # bootstrap over countries for the share of identical references and Jaccard
    rA,pA,_,_=rules(S[a].values,Z['A']); rB,pB,_,_=rules(S[b].values,Z['B']); bs=[];bj=[]
    for _ in range(1000):
        i=rng.choice(n,n,replace=True); bs.append((rA[i]==rB[i]).mean()); bj.append(jac(pA[i],pB[i]))
    r['same_reference_ci']=np.percentile(bs,[2.5,97.5]).tolist(); r['jac_ci']=np.percentile(bj,[2.5,97.5]).tolist(); res[name]=r
out['rules']=res
# null: independent random partitions with the observed cluster-size multisets
nulls={}
for name,(a,b) in {'A4_vs_B5':('cl_A','cl_B'),'k5':('A_k5','B_k5')}.items():
    la,lb=S[a].values,S[b].values; sm=[];jj=[]
    for _ in range(300):
        pa=rng.permutation(la); pb=rng.permutation(lb)
        rA,pA,_,_=rules(pa,Z['A']); rB,pB,_,_=rules(pb,Z['B']); sm.append((rA==rB).mean()); jj.append(jac(pA,pB))
    nulls[name]=dict(same_reference_mean=float(np.mean(sm)),same_reference_ci=np.percentile(sm,[2.5,97.5]).tolist(),jac_mean=float(np.mean(jj)),jac_ci=np.percentile(jj,[2.5,97.5]).tolist())
out['null']=nulls
json.dump(out,open('results/p13_fixed_objectives.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
