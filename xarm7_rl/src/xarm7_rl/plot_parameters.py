"""Render the complete parameter matrix, paired landings, and physical box setup."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


NAMES = {"center_060":"60 g, centered", "center_100":"100 g, centered", "corner_100":"100 g, corner"}
ORDER = ["center_060","center_100","corner_100"]


def cube(ax,center,half,color,alpha):
    vertices=np.array(list(itertools.product([-1,1],repeat=3)))*half+center
    faces=[[vertices[i] for i in range(8) if vertices[i,axis]==center[axis]+sign*half[axis]]
           for axis in range(3) for sign in [-1,1]]
    # Sort each rectangular face around its center to avoid crossed polygons.
    ordered=[]
    for face in faces:
        face=np.asarray(face);c=face.mean(0);axes=np.flatnonzero(np.ptp(face,axis=0)>1e-9)
        angle=np.arctan2(face[:,axes[1]]-c[axes[1]],face[:,axes[0]]-c[axes[0]])
        ordered.append(face[np.argsort(angle)])
    ax.add_collection3d(Poly3DCollection(ordered,facecolors=color,edgecolors="black",alpha=alpha,linewidths=.7))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--results",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();root=Path(args.results);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((root/"table.json").read_text());lookup={(r['policy'],r['physics']):r for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(13,5.1),layout="constrained")
    values=np.array([[100*lookup[p,w]['success_rate'] for w in ORDER] for p in ORDER])
    axes[0].imshow(values,vmin=0,vmax=100,cmap="RdYlGn")
    axes[0].set_xticks(range(3),[NAMES[w] for w in ORDER],rotation=15)
    axes[0].set_yticks(range(3),[NAMES[p] for p in ORDER]);axes[0].set_xlabel("True simulated object");axes[0].set_ylabel("Policy training model / frozen estimate")
    for i,p in enumerate(ORDER):
        for j,w in enumerate(ORDER):
            r=lookup[p,w];axes[0].text(j,i,f"{r['hits']}/{r['episodes']}\n{100*r['success_rate']:g}%",ha="center",va="center",fontsize=13)
    axes[0].set_title("Complete transfer matrix (1 ms, hit radius 10 cm)")
    for w,color in [('center_100','#2678b2'),('corner_100','#d95d39')]:
        data=json.loads((root/f'center_060__on__{w}__dt_0.001'/'summary.json').read_text())['results']
        error=np.array([np.asarray(r['landing']['position'])[:2]-np.asarray(r['target'])[:2] for r in data if r['landing']])
        axes[1].scatter(error[:,0]*100,error[:,1]*100,s=14,alpha=.5,label='Actual '+NAMES[w],color=color)
    axes[1].add_patch(Circle((0,0),10,fill=False,color='green',lw=2,label='Unchanged 10 cm hit radius'))
    axes[1].scatter([0],[0],marker='+',s=100,color='black');axes[1].set_aspect('equal');axes[1].grid(alpha=.25)
    axes[1].set_xlabel('Landing x error (cm)');axes[1].set_ylabel('Landing y error (cm)')
    axes[1].set_title('Same 60 g / center policy: actual mass stays 100 g\nOnly the internal ballast position changes');axes[1].legend(fontsize=8)
    fig.savefig(out/'transfer_matrix.png',dpi=180);fig.savefig(out/'transfer_matrix.svg');plt.close(fig)
    fig=plt.figure(figsize=(11,5.5),layout='constrained')
    for index,w in enumerate(['center_100','corner_100']):
        ax=fig.add_subplot(1,2,index+1,projection='3d');obj=lookup[w,w]['physical_object'];b=obj['ballast']
        cube(ax,np.zeros(3),np.array(obj['size'])*1000,'#8ecae6',.08)
        cube(ax,np.array(b['position'])*1000,np.array(b['half_sizes'])*1000,'#e9b44c',.7)
        c=np.array(obj['com'])*1000;ax.scatter(*c,color='red',s=55,label='True CoM')
        ax.scatter(0,0,0,marker='x',color='blue',s=65,label='Center assumption')
        ax.set(xlim=(-32,32),ylim=(-27,27),zlim=(-37,37),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
        ax.set_box_aspect((60,50,70));ax.view_init(elev=22,azim=-55)
        ax.set_title(f"{NAMES[w]}\nCoM [{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}] mm");ax.legend(fontsize=8)
    fig.suptitle('Same 60 x 50 x 70 mm box: 20 g fixed shell + 80 g fixed metal block\nInternal configuration schematic; exterior collision geometry is identical')
    fig.savefig(out/'ballast_setup.png',dpi=180);fig.savefig(out/'ballast_setup.svg');plt.close(fig)
    for svg in out.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
