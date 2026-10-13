"""Phases 1-4: verification, entity filter, features, coverage, dependency screen."""
import pandas as pd, numpy as np, json
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
raw=pd.read_csv('/mnt/user-data/uploads/owid_energy.csv')
V={}
V['rows'],V['cols']=raw.shape; V['entities']=int(raw.country.nunique()); V['year_min']=int(raw.year.min()); V['year_max']=int(raw.year.max())
V['numeric_cols_excl_year']=int(raw.select_dtypes('number').shape[1]-1)
V['dup_rows']=int(raw.duplicated().sum()); V['dup_keys']=int(raw.duplicated(['country','year']).sum())
iso=raw.iso_code.notna()
V['n_noiso_entities']=int(raw[~iso].country.nunique()); V['n_iso_entities']=int(raw[iso].country.nunique())
d=raw[iso & raw.year.between(2000,2022)].copy()
V['panel_rows_2000_2022']=len(d); V['panel_countries']=int(d.country.nunique())
d['gdp_pc']=d.gdp/d.population
d['ln_gdp_pc']=np.log(d.gdp_pc)
d['ln_energy_pc']=np.log(d.total_energy_per_capita_kwh.where(d.total_energy_per_capita_kwh>0))
d['ln_elecdem_pc']=np.log(d.electricity_demand_per_capita_kwh.where(d.electricity_demand_per_capita_kwh>0))
d['carbon_int']=d.electricity_carbon_intensity_gco2eq_per_kwh
for s,c in [('coal','coal'),('gas','gas'),('oil','oil'),('nuclear','nuclear'),('hydro','hydro'),('wind','wind'),('solar','solar'),('bio','bioenergy')]:
    d[f'{s}_sh']=d[f'{c}_electricity_share_pct']
d['othren_sh']=d.other_renewables_excluding_bioenergy_electricity_share_pct
d['fossil_sh']=d.fossil_fuels_electricity_share_pct
d['renew_sh']=d.renewables_electricity_share_pct
d['lowc_sh']=d.low_carbon_electricity_share_pct
d['netimp_sh']=d.electricity_net_imports_share_of_demand_pct
d['ln1p_coalprod_pc']=np.log1p(d.coal_production_per_capita_kwh)
d['ln1p_oilprod_pc']=np.log1p(d.oil_production_per_capita_kwh)
d['ln1p_gasprod_pc']=np.log1p(d.gas_production_per_capita_kwh)
d['ln_energy_per_gdp']=np.log(d.total_energy_per_gdp_kwh_per_dollar.where(d.total_energy_per_gdp_kwh_per_dollar>0))
d['fossil_primary_sh']=d.fossil_fuels_energy_share_pct
d['lowc_primary_sh']=d.low_carbon_energy_share_pct
src=['coal_sh','gas_sh','oil_sh','nuclear_sh','hydro_sh','wind_sh','solar_sh','bio_sh','othren_sh']
S=d[src]; ok=S.notna().all(axis=1)
P=S.div(S.sum(axis=1),axis=0).where(ok)
d['elec_div']=(-(P*np.log(P.where(P>0))).sum(axis=1,min_count=1))/np.log(len(src))
d['import_exp']=d.netimp_sh.clip(lower=0)
d['energy_int']=d.total_energy_per_gdp_kwh_per_dollar
chk={}
t=d[['ln_energy_per_gdp','ln_energy_pc','ln_gdp_pc']].dropna()
chk['lnEI_minus(lnEpc-lnGDPpc)_maxabs']=float((t.ln_energy_per_gdp-(t.ln_energy_pc-t.ln_gdp_pc)).abs().max())
chk['lnEI_minus(lnEpc-lnGDPpc)_median_abs']=float((t.ln_energy_per_gdp-(t.ln_energy_pc-t.ln_gdp_pc)).abs().median())
t=d[['fossil_sh','lowc_sh']].dropna(); chk['fossil+lowcarbon_mean']=float((t.fossil_sh+t.lowc_sh).mean()); chk['fossil+lowcarbon_sd']=float((t.fossil_sh+t.lowc_sh).std())
t=d[['fossil_sh','coal_sh','gas_sh','oil_sh']].dropna(); chk['fossil_vs_coal+gas+oil_maxabs']=float((t.fossil_sh-t.coal_sh-t.gas_sh-t.oil_sh).abs().max())
t=d[['renew_sh','hydro_sh','wind_sh','solar_sh','bio_sh','othren_sh']].dropna(); chk['renew_vs_components_maxabs']=float((t.renew_sh-t[['hydro_sh','wind_sh','solar_sh','bio_sh','othren_sh']].sum(axis=1)).abs().max())
t=d[src].dropna(); chk['source_share_sum_mean']=float(t.sum(axis=1).mean()); chk['source_share_sum_sd']=float(t.sum(axis=1).std())
V['identity_checks']=chk
cand=['ln_gdp_pc','ln_energy_pc','ln_elecdem_pc','carbon_int','coal_sh','gas_sh','oil_sh','nuclear_sh','hydro_sh','wind_sh','solar_sh','bio_sh','othren_sh','netimp_sh','ln1p_coalprod_pc','ln1p_oilprod_pc','ln1p_gasprod_pc','ln_energy_per_gdp','fossil_sh','renew_sh','lowc_sh','fossil_primary_sh','lowc_primary_sh','elec_div','import_exp']
cov=pd.DataFrame({'coverage_pct':d[cand].notna().mean()*100})
cov['countries_ge80pct_years']=[int((d.groupby('country')[v].apply(lambda s:s.notna().mean())>=0.8).sum()) for v in cand]
cov['missing_pct']=100-cov.coverage_pct
cov.to_csv('results/variable_coverage_panel.csv'); print(cov.round(1).to_string())
d.to_pickle('results/panel.pkl')
pool=['ln_gdp_pc','ln_energy_pc','ln_elecdem_pc','carbon_int','coal_sh','gas_sh','oil_sh','nuclear_sh','hydro_sh','wind_sh','solar_sh','bio_sh','netimp_sh','ln1p_coalprod_pc','ln1p_oilprod_pc','ln1p_gasprod_pc']
R=d[pool].corr(method='spearman'); R.to_csv('results/spearman_pool.csv')
D=1-R.abs().values; np.fill_diagonal(D,0); Z=linkage(squareform(D,checks=False),'average')
cl=fcluster(Z,t=0.20,criterion='distance')
cm=pd.DataFrame({'var':pool,'cluster':cl,'coverage':[cov.loc[v,'coverage_pct'] for v in pool]})
print(cm.sort_values('cluster').to_string())
print([(a,b,round(R.loc[a,b],2)) for i,a in enumerate(pool) for b in pool[i+1:] if abs(R.loc[a,b])>0.7])
json.dump(V,open('results/p1_verification.json','w'),indent=1,default=str); print(json.dumps(V,indent=1,default=str))
