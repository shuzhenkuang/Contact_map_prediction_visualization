#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pysam
import itertools
from model_utils import *
import argparse
import random
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=12)       
plt.rc('axes', titlesize=12) 
plt.rc('axes', labelsize=14) 
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
random.seed(433)  

def pcolormesh_45deg(ax, mat, chrom, start, title=None,markerLabelInterval=0.2, lowLim=0, hiLim=448, *args, **kwargs):
    #Modified from https://stackoverflow.com/questions/12848581/is-there-a-way-to-rotate-a-matplotlib-plot-by-45-degrees
    n = mat.shape[0]
    # create rotation/scaling matrix
    t = np.array([[1,0.5],[-1,0.5]])
    # create coordinate matrix and transform it
    A = np.dot(np.array([(i[1],i[0]) for i in itertools.product(range(n,-1,-1),range(0,n+1,1))]),t)
    im = ax.pcolormesh(A[:,1].reshape(n+1,n+1),A[:,0].reshape(n+1,n+1),np.flipud(mat),*args, **kwargs)
    _ = im.set_rasterized(True)
    _ = ax.set_ylim(0,n)
    _ = ax.spines['right'].set_visible(False)
    _ = ax.spines['top'].set_visible(False)
    _ = ax.spines['left'].set_visible(False)
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.plot([0, n/2], [0, n], 'k-',linewidth=1)
    _ = ax.plot([n/2, n], [n, 0], 'k-',linewidth=1)
    
    end = start + 2**20
    viz_start = start + (32+lowLim)*2048
    viz_end= start +(32+hiLim)*2048
    marker_labels = np.arange(np.floor(start/100000)/10, np.ceil((end)/100000)/10,markerLabelInterval)
    marker_labels = marker_labels[(marker_labels > viz_start/1000000) & (marker_labels < viz_end/1000000)]

    marker_loc = [(x*1000000-(start+(32*2048)))/2048 for x in marker_labels]
    marker_loc_y = [0 for x in marker_labels]
    t = mpl.markers.MarkerStyle(marker='|');
    _ = ax.scatter(x=marker_loc,y=marker_loc_y,marker = t,color='k',s=10);
    for loc,label in zip(marker_loc,marker_labels):
        _ = ax.text(loc, -30, str(round(label,1)), horizontalalignment='center', verticalalignment='center',fontsize=8);
    
    ax.set_aspect(.5)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", "-f", type=str, required=True, help="the genome fasta file")
    parser.add_argument("--model", "-m", type=str, required=True, help="the pretrained Akita model")
    parser.add_argument("--params", "-p", type=str, required=True, help="the parameter file of Akita")
    parser.add_argument("--bedfile", "-b", type=str, required=True, help="the bed file of the mutated region")
    parser.add_argument("--type", "-t", type=str, required=True, help="the mutation type: del or mut")
    parser.add_argument( "--outdir","-o",type=str, required=True, help="the directory to store the output files")
    args = parser.parse_args()

    fasta_open = pysam.Fastafile(args.fasta)
    seqnn_model, params_model = load_Akita(args.model, args.params)
    target_length = params_model['target_length']
    cropping = 32
    target_length_cropped = target_length - 2 * cropping
    with open(args.bedfile,'r') as f:
        for line in f:
            reg = line.strip().split('\t')
            chrom = reg[0]
            start = int(reg[1])
            end = int(reg[2])
            chr_str = "{}_{}_{}".format(chrom, start, end)
            if(args.type=='del'):
                wt_mat,mut_mat,del_idx,region_start,region_stop = make_del_preds(chrom, start, end, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19)
            else:
                wt_mat,mut_mat,region_start,region_stop = make_mut_preds(chrom, start, end, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19)

            fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(12,3))
            pcolormesh_45deg(ax1, wt_mat,chrom,region_start,cmap= 'RdBu_r', vmax=2, vmin=-2)
            pcolormesh_45deg(ax2, mut_mat,chrom,region_start,cmap='RdBu_r', vmax=2, vmin=-2)
            pcolormesh_45deg(ax3, mut_mat-wt_mat,chrom,region_start,cmap= 'PRGn_r', vmax=0.3, vmin=-0.3)
            plt.savefig(f'{args.outdir}/{chr_str}.pdf')

if __name__ == '__main__':
    main()





