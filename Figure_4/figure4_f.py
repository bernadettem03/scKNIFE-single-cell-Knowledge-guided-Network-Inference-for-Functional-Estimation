"""Approved marker-expression UMAP grid, plotted from frozen data."""
import hashlib,json
from pathlib import Path
import h5py,numpy as np
from matplotlib.colors import LinearSegmentedColormap
TITLE_FONT=8.5
GENE_FONT=7
ANNOTATION_FONT=5.5
TICK_FONT=5
POINT_SIZE=.10
SHORT_LABELS=['Endothelial-enriched','T/NK-lineage-enriched','CAF/perivascular','Macrophages','Epithelial','Epithelial']
def draw_marker_panel(fig,rect,path):
 path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['bundle_sha256']
 with h5py.File(path,'r') as f:xy=f['umap'][:];x=f['expression'][:];genes=f['genes'].asstr()[:];vmax=f['vmax'][:]
 assert xy.shape==(69391,2) and x.shape==(69391,6)
 assert list(genes)==['EGFL7','ACAP1','PDGFRB','CD163','KRT7','TRPS1']
 w,h=fig.get_size_inches()*25.4
 def box(a,b,c,d):return [(rect[0]+a*rect[2])/w,(rect[1]+b*rect[3])/h,c*rect[2]/w,d*rect[3]/h]
 fig.text((rect[0]+.8)/w,(rect[1]+rect[3]-1)/h,'F  Representative marker expression',fontsize=TITLE_FONT,fontweight='bold',va='top')
 cmap=LinearSegmentedColormap.from_list('white_purple',['#ffffff','#b997cf','#542788'])
 for k,gene in enumerate(genes):
  row,col=divmod(k,3);left=.012+col*.332;bottom=.13+(1-row)*.41
  ax=fig.add_axes(box(left,bottom,.31,.29));order=np.argsort(x[:,k],kind='stable')
  im=ax.scatter(xy[order,0],xy[order,1],c=x[order,k],cmap=cmap,vmin=0,vmax=vmax[k],s=POINT_SIZE,linewidths=0,rasterized=True)
  ax.set_aspect('equal');ax.set_axis_off()
  fig.text((rect[0]+(left+.155)*rect[2])/w,(rect[1]+(bottom+.345)*rect[3])/h,gene,fontsize=GENE_FONT,fontweight='bold',ha='center')
  fig.text((rect[0]+(left+.155)*rect[2])/w,(rect[1]+(bottom+.315)*rect[3])/h,SHORT_LABELS[k],fontsize=ANNOTATION_FONT,ha='center')
  cax=fig.add_axes(box(left+.065,bottom-.02,.18,.012))
  cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,vmax[k]])
  cb.ax.set_xticklabels(['0',f'{vmax[k]:.1f}']);cb.ax.tick_params(labelsize=TICK_FONT,length=1,pad=1);cb.outline.set_linewidth(.25)
 fig.text((rect[0]+rect[2]/2)/w,(rect[1]+.025*rect[3])/h,'log1p-normalized expression · gene-specific positive-cell P98 scales',fontsize=TICK_FONT,ha='center')
