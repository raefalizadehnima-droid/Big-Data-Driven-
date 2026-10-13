import json,sys; sys.path.insert(0,'code')
from common import *
p4=json.load(open('results/p4_results.json')); p5=json.load(open('results/p5_results.json')); p6=json.load(open('results/p6_results.json')); v1=json.load(open('results/p1_verification.json')); ho=json.load(open('results/p6b_holdout_diag.json'))
R2=pd.read_pickle('results/snapshot2022_ranked.pkl'); Fb=pd.read_pickle('results/snapshot2022_Bfull.pkl')
cov=pd.read_csv('results/variable_coverage_panel.csv',index_col=0); cyr=pd.read_csv('results/coverage_by_year.csv')
def md(df,fn=None):
    s='| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'
    for _,r in df.iterrows(): s+='| '+' | '.join(str(x) for x in r.values)+' |\n'
    return s
T={}
# T1
rows=[('Rows × columns in file',f"{v1['rows']:,} × {v1['cols']}"),('Entities (all) / with ISO code / without',f"{v1['entities']} / {v1['n_iso_entities']} / {v1['n_noiso_entities']}"),('Year range in file',f"{v1['year_min']}–{v1['year_max']}"),('Duplicate rows / duplicate country–year keys',f"{v1['dup_rows']} / {v1['dup_keys']}"),('Analytical panel: ISO entities, 2000–2022',f"{v1['panel_rows_2000_2022']:,} country-years, {v1['panel_countries']} countries"),
 ('Complete country-years on A / B / both (6 snapshot years)',f"{int(cyr.A.sum())} / {int(cyr.B.sum())} / {int(cyr.both.sum())}"),('Countries evaluable in 2022 on A / B',f"{int(cyr.A.iloc[-1])} / {int(cyr.B.iloc[-1])}"),('Countries evaluable in 2022 on A + primary fossil share',f"{int(cyr.Aext.iloc[-1])}"),('Common sample (A ∩ B), country-years / countries',f"{len(pd.read_pickle('results/common_sample.pkl'))} / {pd.read_pickle('results/common_sample.pkl').country.nunique()}")]
T['T1']=md(pd.DataFrame(rows,columns=['Item','Value']))
# T2 final variables
lab={'ln_gdp_pc':('ln GDP per capita','gdp / population','ln(int.-$ per person)'),'ln_energy_pc':('ln energy per capita','total_energy_per_capita_kwh','ln(kWh per person)'),'fossil_sh':('Fossil share of electricity','fossil_fuels_electricity_share_pct','%'),'carbon_int':('Carbon intensity of electricity','electricity_carbon_intensity_gco2eq_per_kwh','gCO₂e/kWh'),'oil_sh':('Oil share of electricity','oil_electricity_share_pct','√(share)'),'ln1p_oilprod_pc':('ln(1+oil production per capita)','oil_production_per_capita_kwh','ln(kWh pc)'),'gas_sh':('Gas share of electricity','gas_electricity_share_pct','√(share)'),'wind_sh':('Wind share of electricity','wind_electricity_share_pct','√(share)'),'bio_sh':('Bioenergy share of electricity','bioenergy_electricity_share_pct','√(share)'),'solar_sh':('Solar share of electricity','solar_electricity_share_pct','√(share)'),'coal_sh':('Coal share of electricity','coal_electricity_share_pct','√(share)'),'ln1p_coalprod_pc':('ln(1+coal production per capita)','coal_production_per_capita_kwh','ln(kWh pc)'),'nuclear_sh':('Nuclear share of electricity','nuclear_electricity_share_pct','√(share)'),'netimp_sh':('Net imports, share of demand','electricity_net_imports_share_of_demand_pct','signed √(share), clipped ±100%')}
rows=[]
for rep,fs in (('A',A_FEATS),('B',B_FEATS)):
    for f in fs: rows.append((rep,lab[f][0],lab[f][1],lab[f][2],f"{cov.loc[f,'coverage_pct']:.1f}"))
T['T2']=md(pd.DataFrame(rows,columns=['Rep.','Feature','OWID source column','Transformation / unit','Coverage 2000–2022 (%)']))
# T3 dependency
chk=v1['identity_checks']
rows=[('ln(energy per GDP) = ln(energy pc) − ln(GDP pc)',f"max |diff| = {chk['lnEI_minus(lnEpc-lnGDPpc)_maxabs']:.1e}",'Exact identity: energy intensity not an independent feature'),('Fossil share + low-carbon share = 100',f"mean {chk['fossil+lowcarbon_mean']:.2f}, SD {chk['fossil+lowcarbon_sd']:.1e}",'Exact closure'),('Fossil share = coal + gas + oil shares',f"max |diff| = {chk['fossil_vs_coal+gas+oil_maxabs']:.1e}",'Exact identity; aggregate removed from B, sources retained'),('Renewables share = hydro+wind+solar+bio+other',f"max |diff| = {chk['renew_vs_components_maxabs']:.1e}",'Exact identity'),('Nine source shares sum to 100',f"mean {chk['source_share_sum_mean']:.2f}, SD {chk['source_share_sum_sd']:.1e}",'Compositional closure; hydro recoverable by difference'),
 ('ln energy pc ~ ln electricity demand pc','ρ = 0.95','One cluster; electricity-only variable dropped'),('ln GDP pc ~ ln energy pc','ρ = 0.94','Same cluster; GDP pc retained in A, dropped from B'),('Carbon intensity ~ hydro share','ρ = −0.81','One cluster; hydro dropped, carbon intensity retained'),('Oil production pc ~ gas production pc','ρ = 0.86','One cluster; gas production dropped')]
T['T3']=md(pd.DataFrame(rows,columns=['Relation','Statistic (2000–2022 panel)','Consequence']))
# T4 dim reduction
rows=[]
for r in 'AB':
    e=p4[f'evr_{r}']; pa=p4[f'pa_eigs_{r}']; ev=np.array(pa['ev']); rows.append((r,len(A_FEATS if r=='A' else B_FEATS),p4[f'npc_parallel_{r}'],' / '.join(f'{x*100:.1f}' for x in e),f"{sum(e)*100:.1f}",f"{ev.sum()**2/(ev**2).sum():.2f}"))
T['T4']=md(pd.DataFrame(rows,columns=['Rep.','Features','PCs retained (parallel analysis)','Explained variance per PC (%)','Cumulative (%)','Effective dimension (participation ratio)']))
# T5 clustering
sel=pd.read_csv('results/cluster_selection.csv'); stb=pd.read_csv('results/cluster_stability.csv'); m=sel.merge(stb,on=['rep','k'])
m['k']=m.k.astype(int); m=m[m.k.isin([2,3,4,5,6,8])]
T['T5']=md(pd.DataFrame({'Rep.':m.rep,'k':m.k,'Silhouette':m.sil.round(3),'Calinski–Harabasz':m.ch.round(0).astype(int),'Davies–Bouldin':m.db.round(3),'Subsample ARI, mean (SD)':[f'{a:.3f} ({b:.3f})' for a,b in zip(m.stab_mean,m.stab_sd)]}))
# T6 agreement
ag=p4['agreement']; rows=[]
for n,l in [('selected(A4,B5)','A (k=4) vs B (k=5)'),('k4','k=4 both'),('k5','k=5 both'),('k6','k=6 both')]:
    a=ag[n]; rows.append((l,f"{a['ARI']:.3f} [{a['ARI_ci'][0]:.3f}, {a['ARI_ci'][1]:.3f}]",f"{a['AMI']:.3f} [{a['AMI_ci'][0]:.3f}, {a['AMI_ci'][1]:.3f}]",f"{a['VI']:.2f}",f"{a['perm_p']:.3f}"))
a=p4['agreement_2022_selected']; rows.append(('A (k=4) vs B (k=5), 2022 only (n=161)',f"{a['ARI']:.3f}",f"{a['AMI']:.3f}",f"{a['VI']:.2f}",'—'))
T['T6']=md(pd.DataFrame(rows,columns=['Comparison','ARI [95% CI]','AMI [95% CI]','Variation of information (nats)','Permutation p']))
# T7 ranking
rm=p5['rank_main']; r5=p5['rank_A_vs_B5']; w=p5['weights']; ci=p5['rank_main_ci']
rows=[('Spearman ρ [95% CI]',f"{rm['spearman']:.3f} [{ci['spearman'][0]:.3f}, {ci['spearman'][1]:.3f}]",f"{r5['spearman']:.3f}"),('Kendall τ [95% CI]',f"{rm['kendall']:.3f} [{ci['kendall'][0]:.3f}, {ci['kendall'][1]:.3f}]",f"{r5['kendall']:.3f}"),('Share of country pairs in reversed order',f"{rm['pair_reversal']:.3f}",f"{r5['pair_reversal']:.3f}"),('Mean / median |rank displacement|',f"{rm['mean_abs_disp']:.1f} / {rm['median_abs_disp']:.0f}",f"{r5['mean_abs_disp']:.1f} / {r5['median_abs_disp']:.0f}"),('90th percentile / maximum displacement',f"{rm['p90_disp']:.0f} / {rm['max_disp']}",f"{r5['p90_disp']:.0f} / {r5['max_disp']}"),('Countries displaced ≥10% of list length',f"{rm['share_disp_ge10pct']*100:.1f}%",f"{r5['share_disp_ge10pct']*100:.1f}%"),('Countries displaced ≥25% of list length',f"{rm['share_disp_ge25pct']*100:.1f}%",f"{r5['share_disp_ge25pct']*100:.1f}%"),('Top-10 / top-20 overlap (count)',f"{rm['top10_overlap']} / {rm['top20_overlap']}",f"{r5['top10_overlap']} / {r5['top20_overlap']}"),('Changed quintile / moved ≥2 quintiles',f"{rm['quintile_change_share']*100:.1f}% / {rm['quintile_change_ge2']*100:.1f}%",f"{r5['quintile_change_share']*100:.1f}% / {r5['quintile_change_ge2']*100:.1f}%")]
T['T7']=md(pd.DataFrame(rows,columns=['Statistic (n = 159, 2022, equal weights)','A (3 obj.) vs B (4 obj.)','A vs B5 (5 obj.)']))
# T8 pareto lists
EA=set(R2.country[R2.frontA==0]); EB=set(R2.country[R2.frontB==0]); both=sorted(EA&EB)
def lst(s,fr): return ', '.join(sorted(s))
rows=[('Efficient in both (n=%d)'%len(both),lst(both,0)),('Efficient only under A (n=%d)'%len(EA-EB),lst(EA-EB,0)),('Efficient only under B (n=%d)'%len(EB-EA),lst(EB-EA,0))]
T['T8a']=md(pd.DataFrame(rows,columns=['Set','Countries']))
pa=p5['pareto']
rows=[('Pareto-efficient countries, A (3 obj.) / B (4 obj.) / B5 (5 obj.)',f"{pa['nA']} / {pa['nB']} / {pa['nB5']}"),('Expected under independent objectives (mean [95% interval]), A / B',f"{pa['null_A_mean']:.1f} [{pa['null_A_ci'][0]:.0f}, {pa['null_A_ci'][1]:.0f}] / {pa['null_B_mean']:.1f} [{pa['null_B_ci'][0]:.0f}, {pa['null_B_ci'][1]:.0f}]"),('Efficient in both A and B; A only; B only',f"{pa['both']}; {pa['onlyA']}; {pa['onlyB']}"),('Jaccard overlap A–B / A–B5',f"{pa['jaccard']:.3f} / {pa['jaccard_B5']:.3f}"),('Number of non-dominated fronts, A / B',f"{pa['n_fronts_A']} / {pa['n_fronts_B']}"),('Efficient in ≥90% of 500 noisy draws, A / B / both',f"{pa['robust_eff_A_ge90']} / {pa['robust_eff_B_ge90']} / {pa['robust_both_ge90']}"),('Mean selection frequency of efficient members, A / B',f"{pa['mean_freq_eff_members_A']:.2f} / {pa['mean_freq_eff_members_B']:.2f}")]
T['T8b']=md(pd.DataFrame(rows,columns=['Statistic','Value']))
sub=pd.DataFrame(pa['B_3obj_subsets']); sub['objs']=sub.objs.apply(lambda x:', '.join(x)); T['T8c']=md(sub.rename(columns={'objs':'B objectives (3-subset)','n_eff':'Efficient countries','overlap_with_A':'Overlap with A-efficient set'}))
robust=sorted(set(R2.country[R2.pfreqA>=0.9])|set(R2.country[R2.pfreqB>=0.9])); T['robust_list']=', '.join(f"{c} (A {R2.set_index('country').pfreqA[c]:.2f}, B {R2.set_index('country').pfreqB[c]:.2f})" for c in robust)
# T9 robustness
cs=p6['cluster_stability_paired']; lv=p6['leave_one_variable_out']; al=p6['normalisation_outlier_alt']; kn=p6['knn_imputation']; rs=p6['rank_stability_subsample']; T9=[]
T9.append(('Cluster: 80% country subsample (100 draws), mean ARI [95% range]',f"{cs['A_mean']:.3f} [{cs['A_ci'][0]:.3f}, {cs['A_ci'][1]:.3f}]",f"{cs['B_mean']:.3f} [{cs['B_ci'][0]:.3f}, {cs['B_ci'][1]:.3f}]"))
T9.append(('Cluster: leave-one-variable-out, mean (min) ARI',f"{lv['A']['mean']:.3f} ({lv['A']['min']:.3f}, drop {lv['A']['argmin']})",f"{lv['B']['mean']:.3f} ({lv['B']['min']:.3f}, drop {lv['B']['argmin']})"))
for k_,l in [('rank','Cluster: rank-based normalisation, ARI'),('raw','Cluster: no winsorisation, ARI'),('winsor5-95','Cluster: winsorise 5–95, ARI'),('drop_top2.5pct_outliers','Cluster: drop top 2.5% outliers, ARI')]: T9.append((l,f"{al['A'][k_]:.3f}",f"{al['B'][k_]:.3f}"))
T9.append(('Cluster: kNN imputation (k=5), ARI on common rows',f"{kn['A']['ARI_vs_baseline_on_common']:.3f} (n set {kn['A']['n_rows_imputed_set']}, {kn['A']['n_rows_with_any_imputation']} imputed)",f"{kn['B']['ARI_vs_baseline_on_common']:.3f} (n set {kn['B']['n_rows_imputed_set']}, {kn['B']['n_rows_with_any_imputation']} imputed)"))
T9.append(('Cluster: out-of-time (train ≤2010, test ≥2015), ARI vs pooled labels',f"{ho['A']['out_of_time_ARI']:.3f} (in-sample {ho['A']['in_sample_ARI']:.3f})",f"{ho['B']['out_of_time_ARI']:.3f} (in-sample {ho['B']['in_sample_ARI']:.3f})"))
T9.append(('Ranking: 80% country subsample (300 draws), mean Spearman ρ [95% range]',f"{rs['A_mean']:.4f} [{rs['A_ci'][0]:.4f}, {rs['A_ci'][1]:.4f}]",f"{rs['B_mean']:.4f} [{rs['B_ci'][0]:.4f}, {rs['B_ci'][1]:.4f}]"))
T9.append(('Ranking: top-10 members retained in subsample (of 10), mean',f"{rs['top10_retained_A']:.2f}",f"{rs['top10_retained_B']:.2f}"))
T9.append(('Ranking: τ vs baseline under 2000 Dirichlet weight draws, mean [95% range]',f"{w['A_within_tau_mean']:.3f} [{w['A_within_tau_ci'][0]:.3f}, {w['A_within_tau_ci'][1]:.3f}]",f"{w['B_within_tau_mean']:.3f} [{w['B_within_tau_ci'][0]:.3f}, {w['B_within_tau_ci'][1]:.3f}]"))
T9.append(('Ranking: mean width of 5–95% rank interval under random weights (positions)',f"{w['A_rank_interval_width_mean']:.1f}",f"{w['B_rank_interval_width_mean']:.1f}"))
T9.append(('Ranking: draws with within-representation τ ≥ cross-representation baseline τ',f"{w['share_withinA_ge_crossbase']*100:.1f}%",f"{w['share_withinB_ge_crossbase']*100:.1f}%"))
T['T9']=md(pd.DataFrame(T9,columns=['Check','Representation A','Representation B']))
# temporal table
tp=p6['temporal']; rows=[]
for r,l in [('A','A (k=4)'),('B','B (k=5)'),('A5','A (k=5)'),('B4','B (k=4)'),('A6','A (k=6)'),('B6','B (k=6)')]:
    t=tp[r]; rows.append((l,t['n_transitions'],f"{t['persistence']:.3f} [{t['persistence_ci'][0]:.3f}, {t['persistence_ci'][1]:.3f}]",f"{t['ari_5yr']:.3f} [{t['ari_5yr_ci'][0]:.3f}, {t['ari_5yr_ci'][1]:.3f}]",f"{t['changed_2000_2022']*100:.1f}% (n={t['n_2000_2022']})"))
T['T10']=md(pd.DataFrame(rows,columns=['Representation','5-year transitions','Persistence [95% CI]','ARI between t and t+5 [95% CI]','Changed state 2000→2022']))
# cluster profile table
pr=pd.read_csv('results/cluster_profiles.csv'); pr=pr.round(1); pr['label']=pr.rep+pr.cluster.astype(str)
T['T11']=md(pr[['label','n','countries','gdp_pc','energy_pc_kwh','fossil_sh','carbon_int','renew_sh','elec_div','netimp']].rename(columns={'label':'Cluster','n':'Country-years','countries':'Countries','gdp_pc':'GDP pc','energy_pc_kwh':'Energy pc (kWh)','fossil_sh':'Fossil el. share (%)','carbon_int':'Carbon int.','renew_sh':'Renewables el. share (%)','elec_div':'Diversity','netimp':'Net imports (%)'}))
# coverage consequence & others
T['cov']=json.dumps(p5['coverage_consequence'],indent=0)
etab=pd.DataFrame(p4['eta2']).round(3); T['eta']=md(etab.reset_index().rename(columns={'index':'indicator'}))
# top/bottom ranks and big movers
R2['disp']=R2.rankA-R2.rankB
big=R2.reindex(R2.disp.abs().sort_values(ascending=False).index).head(8)[['country','rankA','rankB']]; T['movers']=md(big)
top=R2.sort_values('rankA').head(10)[['country','rankA','rankB']]; T['topA']=md(top); top=R2.sort_values('rankB').head(10)[['country','rankA','rankB']]; T['topB']=md(top)
# B-only top20 etc
bo=Fb[(~Fb.evaluable_A)].sort_values('rankB_full').head(5)[['country','rankB_full']]; T['bonly']=md(bo)
json.dump(T,open('results/tables_md.json','w'),indent=1)
for k,v in T.items(): print('=====',k); print(v)
