"""Plot-only C/D replacement; completed 15-cluster data, no scientific reruns."""
from pathlib import Path
import hashlib,json,textwrap
import h5py,numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

POINT_SIZE=.22
ALPHA=.8
FONT=4.8
LABEL_FONT=8.5
STATE_FONT=3.9
DPI=300
D_CLUSTER_FONT=6
D_STATE_FONT=6

def validate_cd(path):
 path=Path(path)
 meta=json.loads(path.with_name('fig4_cd_manifest.json').read_text())
 if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['bundle_sha256']:
  raise ValueError('C/D bundle checksum mismatch')
 with h5py.File(path,'r') as f:
  assert f['umap'].shape==(69391,2) and f['heatmap'].shape==(150,69391)
  assert set(f['cluster'][:])==set(range(15)) and len(f['labels'])==15
 return meta

def draw_updated_panel(fig,rect,label,data_path):
 validate_cd(data_path)
 with h5py.File(data_path,'r') as f:
  data={k:(v.asstr()[:] if v.dtype.kind in 'OSU' else v[:]) for k,v in f.items()}
 w,h=fig.get_size_inches()*25.4
 def box(x,y,bw,bh):
  return [(rect[0]+x*rect[2])/w,(rect[1]+y*rect[3])/h,bw*rect[2]/w,bh*rect[3]/h]
 title='Final cell type and state' if label=='C' else 'Top ten DE genes per H cluster'
 fig.text((rect[0]+.8)/w,(rect[1]+rect[3]-1)/h,f'{label}  {title}',ha='left',va='top',fontsize=LABEL_FONT,fontweight='bold')
 colors=data['colors'];labels=data['labels'];clusters=data['cluster']
 if label=='C':
  ax=fig.add_axes(box(.025,.09,.49,.78));xy=data['umap']
  order=np.random.default_rng(42).permutation(len(xy))
  ax.scatter(xy[order,0],xy[order,1],c=colors[clusters[order]],s=POINT_SIZE,alpha=ALPHA,linewidths=0,rasterized=True)
  ax.set_xticks([]);ax.set_yticks([])
  ax.set_aspect('equal',adjustable='box')
  for side in ['top','right']:ax.spines[side].set_visible(False)
  for side in ['left','bottom']:ax.spines[side].set_linewidth(.45)
  leg=fig.add_axes(box(.535,.055,.46,.85));leg.axis('off')
  handles=[Line2D([0],[0],marker='o',linestyle='none',markerfacecolor=c,markeredgewidth=0,markersize=2.6,
    label=textwrap.fill(str(s),width=34)) for s,c in zip(labels,colors)]
  leg.legend(handles=handles,frameon=False,loc='upper left',fontsize=STATE_FONT,
    handlelength=.65,handletextpad=.3,labelspacing=.18,borderaxespad=0)
  return
 # Existing all-cell heatmap conventions: metadata strips, bwr, fixed -3..3.
 left,bottom,width,height=.13,.22,.73,.55
 hg=data['ordered_cluster'];bounds=np.flatnonzero(hg[1:]!=hg[:-1])+1
 edges=np.r_[0,bounds,len(hg)]
 assert np.array_equal(hg[edges[:-1]],np.arange(15))
 centers=(edges[:-1]+edges[1:]-1)/2
 cluster_ax=fig.add_axes(box(left,.915,width,.02));cluster_ax.set_xlim(-.5,len(hg)-.5)
 cluster_ax.axis('off')
 for k,x in enumerate(centers):
  cluster_ax.text(x,.5,str(k),ha='center',va='center',fontsize=D_CLUSTER_FONT)
 specs=[
  ('Patient',data['patient'],[f'P{i}' for i in range(1,8)],['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#F0E442']),
  ('Site',data['site'],['Bx','Surgery','Recurrence'],['#377EB8','#E41A1C','#984EA3']),
  ('Metastatic',data['metastatic'],['Non-metastatic','Metastatic'],['#4DAF4A','#E41A1C']),
  ('Library',data['library'],['Library1','Library2','Library3','Library4'],['#1F78B4','#FFB000','#33A02C','#E31A1C']),
  ('Final cell type',hg,list(range(15)),colors),
  ('H cluster',hg,list(range(15)),colors)]
 for row,(name,values,cats,palette) in enumerate(specs):
  ax=fig.add_axes(box(left,bottom+height+.014+(5-row)*.023,width,.018))
  lookup={v:i for i,v in enumerate(cats)}
  ax.imshow(np.array([lookup[v] for v in values])[None,:],aspect='auto',interpolation='nearest',
    cmap=mpl.colors.ListedColormap(palette),vmin=0,vmax=len(cats)-1,rasterized=True)
  for b in bounds:ax.axvline(b-.5,color='black',linewidth=.20)
  ax.set_xticks([]);ax.set_yticks([]);ax.set_frame_on(False)
  ax.text(-.008,.5,name,transform=ax.transAxes,ha='right',va='center',fontsize=FONT)
 ax=fig.add_axes(box(left,bottom,width,height))
 im=ax.imshow(data['heatmap'],aspect='auto',interpolation='nearest',cmap='bwr',vmin=-3,vmax=3,rasterized=True)
 for b in bounds:ax.axvline(b-.5,color='black',linewidth=.30)
 for b in range(10,150,10):ax.axhline(b-.5,color='black',linewidth=.25)
 ax.set_yticks([]);ax.set_xticks([])
 ax.text(-.008,.5,'Top 10 ranked genes per H cluster',transform=ax.transAxes,ha='right',va='center',rotation=90,fontsize=FONT)
 side=fig.add_axes(box(.865,bottom,.012,height))
 side.imshow(data['marker_groups'][:,None],aspect='auto',cmap=mpl.colors.ListedColormap(colors),vmin=0,vmax=14,rasterized=True)
 side.set_yticks(np.arange(4.5,150,10),[str(k) for k in range(15)],fontsize=D_CLUSTER_FONT)
 side.yaxis.tick_right();side.tick_params(length=0,pad=1);side.set_xticks([])
 fig.text((rect[0]+.907*rect[2])/w,(rect[1]+(bottom+height/2)*rect[3])/h,
   'Marker-source cluster',fontsize=FONT,rotation=90,ha='center',va='center')
 cax=fig.add_axes(box(.92,bottom+.10,.012,height*.6))
 cb=fig.colorbar(im,cax=cax);cb.set_label('Scaled expression',fontsize=FONT);cb.ax.tick_params(labelsize=FONT,length=1)
 leg=fig.add_axes(box(.13,.025,.85,.16));leg.axis('off')
 handles=[Patch(facecolor=c,label=str(v)) for _,_,cats,palette in specs[:4] for v,c in zip(cats,palette)]
 leg.legend(handles=handles,frameon=False,loc='upper left',ncol=8,fontsize=FONT,handlelength=.9,columnspacing=.6,labelspacing=.2)
 state_ax=fig.add_axes(box(.035,-.64,.96,.64));state_ax.axis('off')
 state_ax.text(0,1,'H cluster → final cell type / state',fontsize=D_STATE_FONT,fontweight='bold',va='top')
 for k,(name,color) in enumerate(zip(labels,colors)):
  col,row=divmod(k,8);x=col*.51;y=.86-row*.12
  state_ax.add_patch(mpl.patches.Rectangle((x,y-.025),.018,.03,facecolor=color,edgecolor='none',transform=state_ax.transAxes))
  state_ax.text(x+.026,y,f'{k}: '+textwrap.fill(str(name),width=46),fontsize=D_STATE_FONT,va='center',linespacing=1.05)
