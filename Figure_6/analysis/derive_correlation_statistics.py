#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr,spearmanr
ROOT=Path(__file__).resolve().parents[1];rows=[]
for filename in ("gpr_gpa_vs_scknife_effects.csv","meia_vs_scknife_effects.csv","meia_pathway_vs_gpr_gpa_effects.csv"):
 d=pd.read_csv(ROOT/"data"/filename)
 for contrast,g in d.groupby("contrast"):
  r=pearsonr(g.x,g.y);rho=spearmanr(g.x,g.y);rows.append({"filename":filename,"contrast":contrast,"n":len(g),"pearson_r":r.statistic,"pearson_p":r.pvalue,"spearman_rho":rho.statistic,"spearman_p":rho.pvalue})
pd.DataFrame(rows).to_csv(ROOT/"data/correlation_statistics.csv",index=False)
