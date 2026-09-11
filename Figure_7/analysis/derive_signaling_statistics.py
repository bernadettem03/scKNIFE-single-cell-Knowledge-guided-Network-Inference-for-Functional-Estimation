#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr
ROOT=Path(__file__).resolve().parents[1];d=pd.read_csv(ROOT/"data/strength_correlation.csv.gz");x=d.cellchat_prob.to_numpy();y=d.scknife_score.to_numpy();lx=np.log10(x);ly=np.log10(y);r=pearsonr(lx,ly);rho=spearmanr(lx,ly);rr=pearsonr(x,y);rrho=spearmanr(x,y);slope,intercept=np.polyfit(lx,ly,1);pd.DataFrame([{"n":len(d),"log_pearson_r":r.statistic,"log_pearson_p":r.pvalue,"log_spearman_rho":rho.statistic,"log_spearman_p":rho.pvalue,"raw_pearson_r":rr.statistic,"raw_pearson_p":rr.pvalue,"raw_spearman_rho":rrho.statistic,"raw_spearman_p":rrho.pvalue,"log_linear_slope":slope,"log_linear_intercept":intercept}]).to_csv(ROOT/"data/signaling_statistics.csv",index=False)
