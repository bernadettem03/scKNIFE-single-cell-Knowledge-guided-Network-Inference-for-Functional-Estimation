#!/usr/bin/env python3
"""Derive the small statistical annotation tables used by Figure 2."""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data'
METHODS={'scKNIFE':'scknife','SPECTRA':'spectra','ExpiMap':'expimap','Slalom':'slalom'}
DATASETS=['bassez_c1','bassez_c2','wu_tnbc']

rows=[]
for dataset in DATASETS:
    frames={name:pd.read_csv(DATA/f'{slug}_{dataset}_umap.csv.gz',dtype=str).set_index('cell_id') for name,slug in METHODS.items()}
    shared=sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    for method,frame in frames.items():
        block=frame.loc[shared]
        truth=block['author_cell_type']; labels=block['leiden_cluster']
        rows.append({'method':method,'dataset':dataset,'n_shared':len(shared),'n_clusters':labels.nunique(),'n_author_types':truth.nunique(),'ari':adjusted_rand_score(truth,labels),'ami':adjusted_mutual_info_score(truth,labels),'nmi':normalized_mutual_info_score(truth,labels)})
agreement=pd.DataFrame(rows)
agreement.to_csv(DATA/'agreement_metrics.csv',index=False)

corr=[]
with h5py.File(DATA/'tahoe_plot_data.h5','r') as bundle:
    for gene in ('TP53','PIK3R1'):
        group=bundle[gene]; x=group['expression'][:]
        for score_name,dataset in [('scKNIFE','bgnmf_score'),('Scanpy','scanpy_score')]:
            y=group[dataset][:]; keep=np.isfinite(x)&np.isfinite(y)
            corr.append({'gene':gene,'score':score_name,'n':int(keep.sum()),'pearson_r':pearsonr(x[keep],y[keep]).statistic,'spearman_rho':spearmanr(x[keep],y[keep]).statistic})
pd.DataFrame(corr).to_csv(DATA/'hallmark_correlations.csv',index=False)

sweeps=pd.read_csv(DATA/'tuning_sweeps.csv')
selected=sweeps.loc[sweeps['tuning_selected'].astype(bool)].copy()
selected.to_csv(DATA/'tuning_selections.csv',index=False)
print('Wrote Figure 2 annotation tables')
