import json,numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'font.family':'DejaVu Sans'})
cA,cB='#D55E00','#0072B2'
df=pd.read_csv('results/controlled_runs.csv'); o=json.load(open('results/p12_controlled.json'))
fd=df[df.cond.str.contains('m=')]
f,axs=plt.subplots(1,3,figsize=(10.4,3.5),gridspec_kw={'width_ratios':[1,1,1.15]})
for ax,met,lab in zip(axs[:2],['ARI','tau'],['ARI with Representation B','Kendall τ with Representation B']):
    for fam,c,mk in (('screened',cB,'o'),('unscreened','#666','s')):
        g=fd[fd.screened==fam].groupby('m')[met]; ms=sorted(fd.m.unique()); med=[g.get_group(m).median() for m in ms]; lo=[g.get_group(m).quantile(.25) for m in ms]; hi=[g.get_group(m).quantile(.75) for m in ms]
        ax.plot(ms,med,mk+'-',color=c,label=f'{fam} subsets (median, IQR)'); ax.fill_between(ms,lo,hi,color=c,alpha=.15)
    a=df[df.cond.str.startswith('A (3')].iloc[0]; ax.scatter([3],[a[met]],marker='*',s=140,color=cA,zorder=5,label='Representation A (3 features)')
    u=df[df.cond=='Unscreened pool (16)'].iloc[0]; ax.scatter([16],[u[met]],marker='D',color='k',zorder=5,label='Unscreened pool (16)')
    ax.set_xlabel('Number of features in the subset'); ax.set_ylabel(lab); ax.set_xticks([3,6,9,12,16]); ax.set_ylim(0,1.02)
axs[0].legend(frameon=False,fontsize=6.5,loc='upper left')
d=o['decomposition']; mets=['ARI','tau','jac']; names=['ARI','Kendall τ','Pareto Jaccard']; comp=[('number','Number of features','#999999'),('screening','Screening (redundancy flag)','#0072B2'),('content','Content (overlap with B)','#E69F00'),('residual','Unexplained','#DDDDDD')]
left=np.zeros(3)
for key,lab,c in comp:
    v=np.array([d[m][key] for m in mets]); axs[2].barh(names,v*100,left=left*100,color=c,label=lab); 
    for i,(x,w) in enumerate(zip(left*100,v*100)):
        if w>6: axs[2].text(x+w/2,i,f'{w:.0f}',ha='center',va='center',fontsize=7)
    left+=v
axs[2].set_xlabel('Share of variance in agreement with B (%)'); axs[2].legend(frameon=False,fontsize=6.5,loc='lower center',bbox_to_anchor=(0.5,-0.55),ncol=2); axs[2].invert_yaxis()
f.tight_layout(); f.savefig('figs/Fig11.png',bbox_inches='tight'); print('ok')
