import json,sys; sys.path.insert(0,'code')
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':300,'font.family':'DejaVu Sans'})
cA,cB='#D55E00','#0072B2'
p4=json.load(open('results/p4_results.json')); p5=json.load(open('results/p5_results.json')); p6=json.load(open('results/p6_results.json')); v1=json.load(open('results/p1_verification.json'))
C=pd.read_pickle('results/common_labeled_disp.pkl'); full=pd.read_pickle('results/panel.pkl')
def save(f,n): f.savefig(f'figs/Fig{n}.png',bbox_inches='tight'); plt.close(f)
# ---- Fig 1 architecture ----
f,ax=plt.subplots(figsize=(9,3.6)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,40)
def box(x,y,w,h,t,c='#EEEEEE'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3',fc=c,ec='#444',lw=0.8)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=6.8)
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=0.9,color='#444'))
sn=full[full.year.isin(YEARS)]
box(1,24,15,12,f"Raw data\n{v1['rows']:,} rows\n{v1['cols']} columns\n{v1['entities']} entities")
box(20,24,15,12,f"Entity filter\nISO-coded only\n{v1['n_iso_entities']} countries\n2000–2022")
box(39,24,17,12,"Dependency audit\nidentities, closure,\nSpearman clusters\n(|ρ|>0.80)")
box(60,29,18,7,"Representation A\n3 conventional features",'#F6D5C2'); box(60,21,18,7,"Representation B\n12 screened features",'#C6DDF0')
box(82,24,16,12,"Latent structure\nparallel analysis\n+ PCA")
box(1,4,17,12,"Classification\nk-means\nstability + agreement")
box(22,4,17,12,"Strategic evaluation\nranking (weighted)\nPareto efficiency")
box(43,4,17,12,"Temporal analysis\ntransitions,\ndispersion, holdout")
box(64,4,17,12,"Robustness\nresampling, weights,\nnormalisation, imputation")
box(85,4,14,12,"Decision\ncomparison\nA versus B",'#E8E8C8')
for a,b in [(16,20),(35,39),(56,60),(78,82)]: arr(a+0.3,30,b-0.3,30)
ax.plot([90,90,9.5,9.5],[23.5,19,19,16.6],color='#444',lw=0.9); arr(9.5,17.4,9.5,16.5); arr(18.3,10,21.7,10); arr(39.3,10,42.7,10); arr(60.3,10,63.7,10); arr(81.3,10,84.7,10)
ax.text(50,39.3,'Same countries, same period, same source dataset — different representations',ha='center',fontsize=8.5,style='italic')
save(f,1)
# ---- Fig 2 coverage ----
f,axs=plt.subplots(1,2,figsize=(9,3.9),gridspec_kw={'width_ratios':[1.25,1]})
cov=pd.read_csv('results/variable_coverage_panel.csv',index_col=0)
sel=A_FEATS+[x for x in B_FEATS if x not in A_FEATS]+['fossil_primary_sh','lowc_primary_sh','elec_div']
lab={'ln_gdp_pc':'GDP per capita (ln)','ln_energy_pc':'Energy per capita (ln)','fossil_sh':'Fossil electricity share','carbon_int':'Carbon intensity of electricity','oil_sh':'Oil share (elec.)','ln1p_oilprod_pc':'Oil production pc','gas_sh':'Gas share (elec.)','wind_sh':'Wind share','bio_sh':'Bioenergy share','solar_sh':'Solar share','coal_sh':'Coal share (elec.)','ln1p_coalprod_pc':'Coal production pc','nuclear_sh':'Nuclear share','netimp_sh':'Net imports share','fossil_primary_sh':'Fossil share, primary energy','lowc_primary_sh':'Low-carbon share, primary energy','elec_div':'Electricity diversity (derived)'}
y=np.arange(len(sel)); vals=[cov.loc[s,'coverage_pct'] for s in sel]
cols=[cA if s in A_FEATS else (cB if s in B_FEATS else '#888888') for s in sel]
axs[0].barh(y,vals,color=cols); axs[0].set_yticks(y); axs[0].set_yticklabels([lab[s] for s in sel],fontsize=7); axs[0].invert_yaxis(); axs[0].set_xlabel('Coverage of ISO country-years, 2000–2022 (%)'); axs[0].set_xlim(0,100)
axs[0].text(98,len(sel)-1,'orange: in A\nblue: in B only\ngrey: not retained',ha='right',va='bottom',fontsize=7)
pyr=[]
for yr in YEARS:
    s=full[full.year==yr]; a=s[A_FEATS].notna().all(axis=1); b=s[B_FEATS].notna().all(axis=1); e=s[A_FEATS+['fossil_primary_sh']].notna().all(axis=1)
    pyr.append((yr,e.sum(),a.sum(),(a&b).sum(),b.sum()))
P=pd.DataFrame(pyr,columns=['y','Aext','A','both','B']); P.to_csv('results/coverage_by_year.csv',index=False)
axs[1].plot(P.y,P.B,'o-',color=cB,label='B (screened)'); axs[1].plot(P.y,P.A,'s-',color=cA,label='A (conventional)'); axs[1].plot(P.y,P.both,'d--',color='k',label='A ∩ B (common sample)'); axs[1].plot(P.y,P.Aext,'^-',color='#888',label='A + primary fossil share')
axs[1].set_ylim(0,230); axs[1].set_xlabel('Year'); axs[1].set_ylabel('Countries with complete features'); axs[1].legend(frameon=False,fontsize=7,loc='center right')
save(f,2)
# ---- Fig 3 dependency ----
R=pd.read_csv('results/spearman_pool.csv',index_col=0); D=1-R.abs().values; np.fill_diagonal(D,0); Z=linkage(squareform(D,checks=False),'average'); order=dendrogram(Z,no_plot=True)['leaves']
Ro=R.iloc[order,order]; f,ax=plt.subplots(figsize=(6.2,5.4)); im=ax.imshow(Ro.values,cmap='RdBu_r',vmin=-1,vmax=1)
nm=[lab.get(c,c) for c in Ro.columns]; nm=[n.replace('oil_sh','Oil share (elec.)') for n in nm]
extra={'ln_elecdem_pc':'Electricity demand pc (ln)','hydro_sh':'Hydro share'}; nm=[extra.get(c,lab.get(c,c)) for c in Ro.columns]
ax.set_xticks(range(len(nm))); ax.set_xticklabels(nm,rotation=60,ha='right',fontsize=7); ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm,fontsize=7)
for i in range(len(nm)):
    for j in range(len(nm)):
        if i!=j and abs(Ro.values[i,j])>0.7: ax.text(j,i,f'{Ro.values[i,j]:.2f}',ha='center',va='center',fontsize=5.5)
plt.colorbar(im,ax=ax,shrink=0.7,label='Spearman ρ (pooled country-years, 2000–2022)'); save(f,3)
# ---- Fig 4 latent ----
f,axs=plt.subplots(2,2,figsize=(8.4,7)); pal=plt.cm.tab10
M={r:build(r,C) for r in 'AB'}
for ax,r,col,k in [(axs[0,0],'A','cl_A',4),(axs[0,1],'B','cl_B',5)]:
    sc=M[r]['sc']; m=(C.year==2022).values
    ax.scatter(sc[:,0],sc[:,1],c=[pal(i) for i in C[col]],s=5,alpha=0.25,lw=0); ax.scatter(sc[m,0],sc[m,1],c=[pal(i) for i in C[col][m]],s=18,edgecolor='k',lw=0.3)
    ax.set_xlabel(f'PC1 ({M[r]["pca"].explained_variance_ratio_[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({M[r]["pca"].explained_variance_ratio_[1]*100:.1f}%)'); ax.set_title(f'Representation {r}: pooled country-years (2022 outlined)',fontsize=8)
for r,c in (('A',cA),('B',cB)):
    ev,thr=M[r]['ev'],M[r]['thr']; x=np.arange(1,len(ev)+1); axs[1,0].plot(x,ev,'o-',color=c,label=f'{r}: observed'); axs[1,0].plot(x,thr,'--',color=c,alpha=.7,label=f'{r}: 95th pct. permuted')
axs[1,0].set_xlabel('Component'); axs[1,0].set_ylabel('Eigenvalue (correlation matrix)'); axs[1,0].legend(frameon=False,fontsize=7); axs[1,0].set_title('Parallel analysis',fontsize=8)
S=C[C.year==2022].reset_index(drop=True); mA,mB=build('A',S),build('B',S); dA,dB=pdist(mA['Z'].values),pdist(mB['Z'].values)
axs[1,1].hexbin(dA,dB,gridsize=40,cmap='Greys',mincnt=1,bins='log'); axs[1,1].set_xlabel('Pairwise distance, A (3-D)'); axs[1,1].set_ylabel('Pairwise distance, B (12-D)')
L=p4['latent']['2022']; axs[1,1].set_title(f"Same country pairs, 2022 (n={L['n']}): Spearman ρ = {L['mantel_rho']:.2f}",fontsize=8)
f.tight_layout(); save(f,4)
# ---- Fig 5 clusters ----
sel_=pd.read_csv('results/cluster_selection.csv'); stb=pd.read_csv('results/cluster_stability.csv')
f,axs=plt.subplots(1,3,figsize=(9.6,3.2))
for r,c in (('A',cA),('B',cB)):
    t=sel_[sel_.rep==r]; axs[0].plot(t.k,t.sil,'o-',color=c,label=r); s=stb[stb.rep==r]; axs[1].errorbar(s.k,s.stab_mean,yerr=s.stab_sd,fmt='o-',color=c,capsize=2,label=r)
    axs[2].plot(t.k,t.db,'o-',color=c,label=r)
axs[0].set_xlabel('k'); axs[0].set_ylabel('Silhouette'); axs[1].set_xlabel('k'); axs[1].set_ylabel('Subsample stability (mean ARI ± SD)'); axs[2].set_xlabel('k'); axs[2].set_ylabel('Davies–Bouldin (lower is better)')
for a in axs: a.legend(frameon=False)
axs[0].axvline(4,color=cA,ls=':',lw=1); axs[0].axvline(5,color=cB,ls=':',lw=1); f.tight_layout(); save(f,5)
# ---- Fig 6 agreement ----
ct=pd.read_csv('results/contingency_selected.csv',index_col=0); f,axs=plt.subplots(1,2,figsize=(9,3.6),gridspec_kw={'width_ratios':[1,1.1]})
rn=ct.div(ct.sum(axis=1),axis=0); im=axs[0].imshow(rn.values,cmap='Blues',vmin=0,vmax=1)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]): axs[0].text(j,i,f'{ct.values[i,j]}',ha='center',va='center',fontsize=8,color='white' if rn.values[i,j]>0.55 else 'k')
axs[0].set_xticks(range(ct.shape[1])); axs[0].set_xticklabels([f'B{j}' for j in ct.columns]); axs[0].set_yticks(range(ct.shape[0])); axs[0].set_yticklabels([f'A{i}' for i in ct.index]); axs[0].set_title('Country-years by A-cluster (rows) and B-cluster (columns); colour = row share',fontsize=7.5)
ag=p4['agreement']; names=['selected(A4,B5)','k4','k5','k6']; xl=['A4 vs B5','k=4','k=5','k=6']
v=[ag[n]['ARI'] for n in names]; lo=[ag[n]['ARI']-ag[n]['ARI_ci'][0] for n in names]; hi=[ag[n]['ARI_ci'][1]-ag[n]['ARI'] for n in names]
axs[1].bar(xl,v,yerr=[lo,hi],color='#777',capsize=3); sA=p6['cluster_stability_paired']['A_mean']; sB=p6['cluster_stability_paired']['B_mean']
axs[1].axhline(sA,color=cA,ls='--',label=f'within-A subsample ARI ({sA:.2f})'); axs[1].axhline(sB,color=cB,ls='--',label=f'within-B subsample ARI ({sB:.2f})'); axs[1].set_ylim(0,1.05); axs[1].set_ylabel('Adjusted Rand index (A vs B labels)'); axs[1].legend(frameon=False,fontsize=7,loc='center right'); axs[1].set_title('Cross-representation agreement (95% bootstrap CI over countries)',fontsize=7.5)
f.tight_layout(); save(f,6)
# ---- Fig 7 rank ----
R2=pd.read_pickle('results/snapshot2022_ranked.pkl'); tc,ta,tb=[np.load(f'results/{n}.npy') for n in ('tau_cross_weights','tau_withinA','tau_withinB')]
f,axs=plt.subplots(1,3,figsize=(10,3.4)); n=len(R2)
axs[0].scatter(R2.rankA,R2.rankB,s=9,color='#444',alpha=.7); axs[0].plot([1,n],[1,n],'k--',lw=.7); axs[0].set_xlabel('Rank under A (1 = best)'); axs[0].set_ylabel('Rank under B (1 = best)'); axs[0].set_title(f"Equal weights, n={n}; Kendall τ = {p5['rank_main']['kendall']:.2f}",fontsize=8)
dd=(R2.rankA-R2.rankB).abs(); axs[1].hist(dd,bins=20,color='#777'); axs[1].axvline(0.1*n,color='r',ls='--',lw=1); axs[1].set_xlabel('|rank displacement|'); axs[1].set_ylabel('Countries'); axs[1].set_title(f"{p5['rank_main']['share_disp_ge10pct']*100:.0f}% moved ≥10% of list (dashed)",fontsize=8)
bins=np.linspace(-0.2,1,41); axs[2].hist(tc,bins,alpha=.55,color='#555',label='A vs B (random weights)'); axs[2].hist(ta,bins,alpha=.5,color=cA,label='A: baseline vs random weights'); axs[2].hist(tb,bins,alpha=.5,color=cB,label='B: baseline vs random weights')
axs[2].axvline(p5['rank_main']['kendall'],color='k',ls='--',lw=1.2); axs[2].set_xlabel('Kendall τ'); axs[2].set_ylabel('Weight draws (of 2000)'); axs[2].legend(frameon=False,fontsize=6.5,loc='upper left'); axs[2].set_title('Weights vs representation (dashed: equal-weight A vs B)',fontsize=8)
f.tight_layout(); save(f,7)
# ---- Fig 8 Pareto ----
f,axs=plt.subplots(1,3,figsize=(10.4,3.6)); effA=R2.frontA==0; effB=R2.frontB==0
cls=np.where(effA&effB,'both',np.where(effA,'A only',np.where(effB,'B only','neither'))); colmap={'both':'#000','A only':cA,'B only':cB,'neither':'#CCC'}
for ax,xk,yk,xl,yl,logx in [(axs[0],'total_energy_per_capita_kwh','fossil_sh','Energy per capita (kWh, log)','Fossil share of electricity (%)',True),(axs[1],'carbon_int','elec_div','Carbon intensity (gCO₂e/kWh)','Electricity diversity (normalised entropy)',False)]:
    for c_ in ['neither','A only','B only','both']:
        m=cls==c_; ax.scatter(R2[xk][m],R2[yk][m],s=14 if c_!='neither' else 8,color=colmap[c_],label=c_,alpha=.9 if c_!='neither' else .6,lw=0)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if logx: ax.set_xscale('log')
axs[0].legend(frameon=False,fontsize=7,title='Pareto-efficient in',title_fontsize=7); axs[0].set_title('Projection of A objectives',fontsize=8); axs[1].set_title('Projection of B objectives',fontsize=8)
pa=p5['pareto']; lbl=['A (3 obj.)','B (4 obj.)','both','robust A\n(≥90%)','robust B\n(≥90%)','robust both']; vals=[pa['nA'],pa['nB'],pa['both'],pa['robust_eff_A_ge90'],pa['robust_eff_B_ge90'],pa['robust_both_ge90']]
axs[2].bar(lbl,vals,color=[cA,cB,'#000',cA,cB,'#000'],alpha=.85); axs[2].set_ylabel('Countries'); axs[2].tick_params(axis='x',labelsize=6.5)
for i,v_ in enumerate(vals): axs[2].text(i,v_+.3,str(v_),ha='center',fontsize=7)
axs[2].set_title('Efficient sets and robustness to 5% noise',fontsize=8); f.tight_layout(); save(f,8)
# ---- Fig 9 temporal ----
f,axs=plt.subplots(1,3,figsize=(10,3.3))
for ax,r,c,tit in [(axs[0],'A',cA,'A'),(axs[1],'B',cB,'B')]:
    M_=pd.read_csv(f'results/transition_matrix_{r}.csv',index_col=0); im=ax.imshow(M_.values,cmap='Greys',vmin=0,vmax=1)
    for i in range(M_.shape[0]):
        for j in range(M_.shape[1]): ax.text(j,i,f'{M_.values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='w' if M_.values[i,j]>.5 else 'k')
    ax.set_xticks(range(M_.shape[1])); ax.set_yticks(range(M_.shape[0])); ax.set_xlabel('State at t+5'); ax.set_ylabel('State at t'); T_=p6['temporal'][r]; ax.set_title(f"{tit}: 5-year transitions (n={T_['n_transitions']}); persistence {T_['persistence']:.2f}",fontsize=7.5)
for r,c in (('A',cA),('B',cB)):
    by=p6['dispersion'][r]['by_year']; axs[2].plot([int(k) for k in by],list(by.values()),'o-',color=c,label=f"{r}: Δ={p6['dispersion'][r]['delta_2000_2022']:+.3f}")
axs[2].set_xlabel('Year'); axs[2].set_ylabel('Mean RMS distance to pooled centroid'); axs[2].legend(frameon=False); axs[2].set_title('Dispersion of the common sample',fontsize=8); f.tight_layout(); save(f,9)
# ---- Fig 10 robustness ----
cs=p6['cluster_stability_paired']; lv=p6['leave_one_variable_out']; al=p6['normalisation_outlier_alt']; kn=p6['knn_imputation']; ho=json.load(open('results/p6b_holdout_diag.json'))
rows=[('Subsample (80% of countries), mean ARI',cs['A_mean'],cs['B_mean']),('Leave-one-variable-out, mean ARI',lv['A']['mean'],lv['B']['mean']),('Rank-based normalisation, ARI',al['A']['rank'],al['B']['rank']),('No winsorisation, ARI',al['A']['raw'],al['B']['raw']),('Winsorise 5–95, ARI',al['A']['winsor5-95'],al['B']['winsor5-95']),('Drop top 2.5% outliers, ARI',al['A']['drop_top2.5pct_outliers'],al['B']['drop_top2.5pct_outliers']),('kNN imputation (k=5), ARI',kn['A']['ARI_vs_baseline_on_common'],kn['B']['ARI_vs_baseline_on_common']),('Out-of-time (train ≤2010, test ≥2015), ARI',ho['A']['out_of_time_ARI'],ho['B']['out_of_time_ARI'])]
f,axs=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1.5,1]}); y=np.arange(len(rows))[::-1]
for i,(n_,a,b) in enumerate(rows): axs[0].plot([a,b],[y[i]]*2,color='#BBB',lw=1,zorder=1)
axs[0].scatter([r[1] for r in rows],y,color=cA,label='A',zorder=2,s=28); axs[0].scatter([r[2] for r in rows],y,color=cB,label='B',zorder=2,s=28)
axs[0].set_yticks(y); axs[0].set_yticklabels([r[0] for r in rows],fontsize=7); axs[0].set_xlim(0,1.05); axs[0].set_xlabel('Agreement with baseline cluster labels (ARI)'); axs[0].legend(frameon=False,loc='lower left'); axs[0].set_title('Cluster robustness',fontsize=8)
rs=pd.read_csv('results/rank_norm_sensitivity.csv'); rs=rs[rs.norm!='minmax']; yy=np.arange(len(rs))[::-1]
axs[1].scatter(rs.A_vs_base_rho,yy,color=cA,s=28); axs[1].scatter(rs.B_vs_base_rho,yy,color=cB,s=28); axs[1].scatter(rs.cross_rho,yy,color='k',marker='x',s=28,label='A vs B (same treatment)')
axs[1].set_yticks(yy); axs[1].set_yticklabels(rs.norm,fontsize=7); axs[1].set_xlim(0.5,1.02); axs[1].set_xlabel('Spearman ρ of rankings'); axs[1].legend(frameon=False,fontsize=6.5,loc='lower left'); axs[1].set_title('Ranking: normalisation and aggregation',fontsize=8)
f.tight_layout(); save(f,10); print('done')
