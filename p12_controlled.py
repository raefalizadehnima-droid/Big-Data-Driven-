"""Controlled disentangling of feature content, feature number and screening (fixed sample, fixed objectives, fixed pipeline)."""
import json,sys,itertools,time; sys.path.insert(0,'code')
from common import *
from scipy.stats import kendalltau
SEEDX=SEED+11; rng=np.random.default_rng(SEEDX)
POOL=['ln_gdp_pc','ln_energy_pc','ln_elecdem_pc','carbon_int','coal_sh','gas_sh','oil_sh','nuclear_sh','hydro_sh','wind_sh','solar_sh','bio_sh','netimp_sh','ln1p_coalprod_pc','ln1p_oilprod_pc','ln1p_gasprod_pc']
GROUPS=[['ln_gdp_pc','ln_energy_pc','ln_elecdem_pc'],['carbon_int','hydro_sh'],['ln1p_oilprod_pc','ln1p_gasprod_pc']]+[[v] for v in ['coal_sh','gas_sh','oil_sh','nuclear_sh','wind_sh','solar_sh','bio_sh','netimp_sh','ln1p_coalprod_pc']]
assert len(GROUPS)==12 and sum(len(g) for g in GROUPS)==16
S=snapshot_rows(); ok=S[POOL].notna().all(axis=1); C=S[ok].reset_index(drop=True)
need=['fossil_sh','energy_int','total_energy_per_capita_kwh','carbon_int','elec_div','import_exp']
y22=(C.year==2022).values; C22=C[y22]; okobj=C22[need].notna().all(axis=1).values
idx22=np.where(y22)[0][okobj]          # row positions (in C) of 2022 objective-complete countries
# fixed objective set O* (computed from raw data, identical for every representation)
OBJ=[('carbon_int',-1),('elec_div',1),('import_exp',-1),('total_energy_per_capita_kwh',1),('energy_int',-1)]
F=np.column_stack([C.iloc[idx22][c].values*s for c,s in OBJ]).astype(float)
a,b=np.nanpercentile(F,2.5,axis=0),np.nanpercentile(F,97.5,axis=0); Fn=(np.clip(F,a,b)-np.clip(F,a,b).min(0)); Fn=Fn/(np.clip(F,a,b).max(0)-np.clip(F,a,b).min(0))
score=Fn.mean(1); leaders=np.argsort(-score)[:10]; nonlead=np.setdiff1d(np.arange(len(F)),leaders)
def nd(X):
    return np.array([not ((X>=X[i]).all(1)&(X>X[i]).any(1)).any() for i in range(len(X))])
def fit(feats,k=5,nperm=100,mode='B',seed=SEED):
    X=tf(C,feats,mode); Z=zscore(winsor(X)); 
    ev,thr,npc=parallel_analysis(Z.values,nperm=nperm,seed=seed); pca=PCA(n_components=max(npc,2),random_state=SEED).fit(Z.values)
    sc=pca.transform(Z.values); lab=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(sc).labels_
    Zs=Z.values[idx22]; D=np.sqrt(((Zs[:,None,:]-Zs[None,leaders,:])**2).sum(2)).mean(1)   # mean distance to the 10 leaders, own feature space
    feas=pd.Series(D[nonlead]).rank().values
    l22=lab[idx22]; par=np.zeros(len(F),bool)
    for c in np.unique(l22):
        m=np.where(l22==c)[0]; par[m[nd(F[m])]]=True
    return dict(lab=lab,feas=feas,par=par,npc=int(npc))
def compare(r,ref):
    return dict(ARI=adjusted_rand_score(ref['lab'],r['lab']),tau=kendalltau(ref['feas'],r['feas'])[0],jac=(ref['par']&r['par']).sum()/max(1,(ref['par']|r['par']).sum()))
ref=fit(B_FEATS); refA5=fit(A_FEATS,k=5,mode='A'); refA4=fit(A_FEATS,k=4,mode='A')
out={'n_rows':len(C),'n_countries_2022':int(len(idx22)),'n_leaders':10,'n_nonleaders':int(len(nonlead)),'ref_pareto_size':int(ref['par'].sum()),'A5_pareto_size':int(refA5['par'].sum())}
rows=[]
def add(cond,feats,r,m=None,screened=None):
    c=compare(r,ref); gmap={v:i for i,g in enumerate(GROUPS) for v in g}; c.update(cond=cond,m=m if m is not None else len(feats),screened=screened,overlap=len(set(feats)&set(B_FEATS))/len(set(feats)|set(B_FEATS)),groups=len({gmap.get(v,v) for v in feats}),feats='|'.join(feats)); rows.append(c)
add('A (3 conventional features), k=5',A_FEATS,refA5,screened='A'); 
cA4=compare(refA4,ref); out['A_k4_vs_B']=cA4
# (a) screening effect: full unscreened pool
add('Unscreened pool (16)',POOL,fit(POOL),screened='unscreened')
# (b) content inside screening: the 12 alternative one-per-group sets (B itself is one of them)
alt=[]
for combo in itertools.product(*GROUPS):
    if set(combo)==set(B_FEATS): continue
    alt.append(list(combo))
for f in alt: add('Screened alternatives (12 groups)',f,fit(f),screened='screened')
t0=time.time()
# (c) factorial: m in {3,6,9,12} x {screened, unscreened}, N draws each
N=100
for m in (3,6,9,12):
    for fam in ('screened','unscreened'):
        for _ in range(N):
            if fam=='screened':
                gi=rng.choice(12,m,replace=False); f=[rng.choice(GROUPS[g]) for g in gi]
            else: f=list(rng.choice(POOL,m,replace=False))
            add(f'{fam} m={m}',f,fit(f),m=m,screened=fam)
        print(m,fam,round(time.time()-t0),flush=True)
df=pd.DataFrame(rows); df.to_csv('results/controlled_runs.csv',index=False)
# summary by condition
def q(x): return f"{x.median():.3f} [{x.quantile(.25):.3f}, {x.quantile(.75):.3f}]"
summ=df.groupby('cond').agg(n=('ARI','size'),ARI=('ARI',q),tau=('tau',q),jac=('jac',q)).reset_index(); summ.to_csv('results/controlled_summary.csv',index=False); print(summ.to_string())
# variance decomposition (LMG / Shapley over blocks) for the factorial design
fd=df[df.cond.str.contains(r'm=')].copy(); fd['scr']=(fd.screened=='screened').astype(float)
def design(blocks):
    cols=[np.ones(len(fd))]
    if 'm' in blocks:
        for mv in (6,9,12): cols.append((fd.m==mv).astype(float).values)
    if 'screen' in blocks: cols.append(fd.scr.values)
    if 'content' in blocks: cols.append(fd.overlap.values)
    return np.column_stack(cols)
def r2(blocks,y):
    X=design(blocks); beta=np.linalg.lstsq(X,y,rcond=None)[0]; res=y-X@beta; return 1-res.var()/y.var() if blocks else 0.0
dec={}
for met in ('ARI','tau','jac'):
    y=fd[met].values.astype(float); B_=['m','screen','content']; sh={b:0.0 for b in B_}
    for perm in itertools.permutations(B_):
        cur=[]; prev=r2(cur,y)
        for b in perm:
            cur=cur+[b]; new=r2(cur,y); sh[b]+=(new-prev)/6; prev=new
    dec[met]=dict(R2_total=r2(B_,y),number=sh['m'],screening=sh['screen'],content=sh['content'],residual=1-r2(B_,y))
fd['redund']=fd.m-fd.groups
def design2(blocks):
    cols=[np.ones(len(fd))]
    if 'm' in blocks:
        for mv in (6,9,12): cols.append((fd.m==mv).astype(float).values)
    if 'redund' in blocks: cols.append(fd.redund.values.astype(float))
    if 'content' in blocks: cols.append(fd.overlap.values)
    return np.column_stack(cols)
def r2b(blocks,y):
    X=design2(blocks); beta=np.linalg.lstsq(X,y,rcond=None)[0]; res=y-X@beta; return 1-res.var()/y.var() if blocks else 0.0
dec2={}
for met in ('ARI','tau','jac'):
    y=fd[met].values.astype(float); B_=['m','redund','content']; sh={b:0.0 for b in B_}
    for perm in itertools.permutations(B_):
        cur=[]; prev=r2b(cur,y)
        for b in perm:
            cur=cur+[b]; new=r2b(cur,y); sh[b]+=(new-prev)/6; prev=new
    dec2[met]=dict(R2_total=r2b(B_,y),number=sh['m'],redundancy=sh['redund'],content=sh['content'],residual=1-r2b(B_,y))
out['decomposition_redundancy']=dec2; print(json.dumps(dec2,indent=1))
# two-factor view without the reference-overlap term (number and screening only)
dec3={}
for met in ('ARI','tau','jac'):
    y=fd[met].values.astype(float); B_=['m','screen']; sh={b:0.0 for b in B_}
    for perm in itertools.permutations(B_):
        cur=[]; prev=r2(cur,y)
        for b in perm:
            cur=cur+[b]; new=r2(cur,y); sh[b]+=(new-prev)/2; prev=new
    dec3[met]=dict(R2_total=r2(B_,y),number=sh['m'],screening=sh['screen'])
out['decomposition_no_content']=dec3
out['decomposition']=dec; print(json.dumps(dec,indent=1))
# marginal means by m and screening
mm=fd.groupby(['m','screened'])[['ARI','tau','jac']].median().round(3); mm.to_csv('results/controlled_by_m.csv'); print(mm)
# k-means seed noise floor for the reference itself
seeds=[fit(B_FEATS,seed=SEEDX+i) for i in range(5)]
out['seed_floor_ARI']=[adjusted_rand_score(ref['lab'],s['lab']) for s in seeds]
out['n_alt_screened']=len(alt)
json.dump(out,open('results/p12_controlled.json','w'),indent=1,default=float)
