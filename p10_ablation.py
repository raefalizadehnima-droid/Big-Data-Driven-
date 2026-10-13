import json,sys; sys.path.insert(0,'code')
import common as cm
from common import *
from scipy.stats import spearmanr
C=pd.read_pickle('results/common_labeled.pkl').reset_index(drop=True)
POOL=['ln_gdp_pc','ln_energy_pc','ln_elecdem_pc','carbon_int','coal_sh','gas_sh','oil_sh','nuclear_sh','hydro_sh','wind_sh','solar_sh','bio_sh','netimp_sh','ln1p_coalprod_pc','ln1p_oilprod_pc','ln1p_gasprod_pc']
ok=C[POOL].notna().all(axis=1); Cu=C[ok].reset_index(drop=True); out={'n_rows_unscreened':int(ok.sum()),'n_rows_common':len(C)}
rng=np.random.default_rng(SEED+7)
def run(feats,rows,k,label):
    old=cm.B_FEATS[:]; cm.B_FEATS[:]=feats
    m=cm.build('B',rows); sc=m['sc']; lab=km(sc,k).labels_; ev=m['ev']
    st=[]; cs=rows.country.unique()
    for _ in range(60):
        s=rng.choice(cs,int(0.8*len(cs)),replace=False); mask=rows.country.isin(s).values; mm=cm.build('B',rows[mask]); st.append(adjusted_rand_score(lab[mask],km(mm['sc'],k).labels_))
    cm.B_FEATS[:]=old
    return dict(label=label,n_features=len(feats),npc=int(m['npc']),eff_dim=float(ev.sum()**2/(ev**2).sum()),silhouette=float(silhouette_score(sc,lab)),stab_mean=float(np.mean(st)),stab_sd=float(np.std(st))),lab
rs=[];labs={}
for nm,f in (('B screened (12)',B_FEATS),('Unscreened pool (16)',POOL),('A conventional (3)',None)):
    if f is None: continue
    r,l=run(f,Cu,5,nm); rs.append(r); labs[nm]=l
# A on same rows
mA=cm.build('A',Cu); lA=km(mA['sc'],4).labels_; evA=mA['ev']
stA=[]; cs=Cu.country.unique()
for _ in range(60):
    s=rng.choice(cs,int(0.8*len(cs)),replace=False); mask=Cu.country.isin(s).values; stA.append(adjusted_rand_score(lA[mask],km(cm.build('A',Cu[mask])['sc'],4).labels_))
rs.append(dict(label='A conventional (3), k=4',n_features=3,npc=int(mA['npc']),eff_dim=float(evA.sum()**2/(evA**2).sum()),silhouette=float(silhouette_score(mA['sc'],lA)),stab_mean=float(np.mean(stA)),stab_sd=float(np.std(stA))))
out['table']=rs
out['ARI_screened_vs_unscreened_k5']=adjusted_rand_score(labs['B screened (12)'],labs['Unscreened pool (16)'])
out['ARI_unscreened_vs_A']=adjusted_rand_score(labs['Unscreened pool (16)'],lA); out['ARI_screened_vs_A']=adjusted_rand_score(labs['B screened (12)'],lA)
# distance-structure: Mantel-like rank corr on 2022 rows
S=Cu[Cu.year==2022].reset_index(drop=True); from scipy.spatial.distance import pdist
old=cm.B_FEATS[:]; cm.B_FEATS[:]=B_FEATS; dS=pdist(cm.build('B',S)['Z'].values); cm.B_FEATS[:]=POOL; dU=pdist(cm.build('B',S)['Z'].values); cm.B_FEATS[:]=old; dA=pdist(cm.build('A',S)['Z'].values)
out['dist_rho_screened_vs_unscreened_2022']=float(spearmanr(dS,dU)[0]); out['dist_rho_unscreened_vs_A_2022']=float(spearmanr(dU,dA)[0]); out['dist_rho_screened_vs_A_2022']=float(spearmanr(dS,dA)[0]); out['n_2022']=len(S)
# Does ranking change with unscreened objectives? objectives are not feature-based -> not applicable; state.
json.dump(out,open('results/p10_ablation.json','w'),indent=1,default=float); print(json.dumps(out,indent=1))
