#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
from cooltools.lib.numutils import set_diag
from basenji import seqnn
import scipy
import json
random.seed(433)

def one_hot_encode(sequence: str,
                alphabet: str = 'ACGT',
                neutral_alphabet: str = 'N',
                neutral_value = 0,
                dtype=np.float32) -> np.ndarray:
    '''Faster One-hot encode sequence. From Enformer paper'''
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]

def load_Akita(model_file,params_file):
    with open(params_file) as params_open:
        params = json.load(params_open)
        params_model = params['model']
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file,0)
    seqnn_model.build_ensemble(True, [-3,-1,0,1,3])
    return seqnn_model.model, params_model

def from_upper_triu(vector_repr, matrix_len, num_diags):
    z = np.zeros((matrix_len,matrix_len))
    triu_tup = np.triu_indices(matrix_len,num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags+1,num_diags):
        set_diag(z, np.nan, i)
    return z + z.T

def interp_all_nans(a_init, pad_zeros=True):
    init_shape = np.shape(a_init)
    if len(init_shape) == 2 and init_shape[0] != 1 and init_shape[1] !=1:
        if pad_zeros == True:
            a = np.zeros((init_shape[0]+2,init_shape[1]+2))
            a[1:-1,1:-1] = a_init
        else:
            a = a_init
        x, y = np.indices(a.shape)
        interp = np.array(a)
        interp[np.isnan(interp)] = scipy.interpolate.griddata(
                 (x[~np.isnan(a)], y[~np.isnan(a)]), 
                  a[~np.isnan(a)],        
                 (x[np.isnan(a)], y[np.isnan(a)]) ,method='linear')
        if pad_zeros == True:
            return interp[1:-1,1:-1]
        else:
            return interp

def makeDel_symmetric(chrm, seq_start, seq_end, del_start, del_end, fasta_open, half_patch_size=2**19):
    del_len = del_end-del_start
    seq_start_del, seq_stop_del =  seq_start-del_len//2,   (seq_end+del_len//2)
    
    if (seq_stop_del - seq_start_del - del_len) != (2*half_patch_size):
        to_add = 2*half_patch_size - (seq_stop_del - seq_start_del - del_len)
        seq_stop_del += to_add
    seq = fasta_open.fetch(chrm, seq_start_del, seq_stop_del).upper()
    print("The sym-padded deletion window starts at {} and ends at {}.".format(seq_start_del, seq_stop_del))

    seq_del = one_hot_encode(seq)
    seq_del  = np.vstack((seq_del[ :(del_start-seq_start_del),:],
                          seq_del[(del_end-seq_start_del):, :] ))
    return seq_del

def predict_wt(region_chr, region_start, region_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19):
    
    if region_stop-region_start != (2*half_patch_size):
        to_add = 2*half_patch_size - (region_stop - region_start)
        region_stop += to_add
    
    seq = fasta_open.fetch( region_chr, region_start, region_stop ).upper()
    seq_1hot = one_hot_encode(seq)
    
    pred_targets = seqnn_model.predict(np.expand_dims(seq_1hot,0) )    
    mat = from_upper_triu(pred_targets[0,:,0],target_length_cropped,2)
    mat_denan = interp_all_nans(mat)
    return mat_denan

def predict_del(region_chr, region_start, region_stop, del_start, del_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19):
    seq_1hot = makeDel_symmetric(region_chr, region_start, region_stop, del_start, del_stop, fasta_open, half_patch_size)
    pred_targets = seqnn_model.predict(np.expand_dims(seq_1hot,0) )    
    sym_mat = from_upper_triu(pred_targets[0,:,0],target_length_cropped,2)
    sym_mat_denan = interp_all_nans(sym_mat)
    return sym_mat_denan

def mask_mat(mat_wt, start_wt, stop_wt, mat_del, del_start, del_stop, target_length_cropped):
    nrow = mat_wt.shape[0]
    bp_per_pix = (stop_wt - start_wt) / nrow
    
    pix_coords = []
    rows_to_mask = []
    for j in range(0,nrow):
        pix_start = bp_per_pix*j + start_wt
        pix_stop = bp_per_pix*(j+1) + start_wt
        pix_coords.append(tuple([pix_start, pix_stop]))

        if pix_stop > del_start and pix_start < del_stop:
            rows_to_mask.append(j)
            
    #wt mask
    wt_masked = mat_wt.copy()
    for j in rows_to_mask:
        wt_masked[j,:] = np.nan
        wt_masked[:,j] = np.nan
    
    #del mask
    del_masked = mat_del.copy()    
    del_masked = del_masked[int(len(rows_to_mask)/2):, int(len(rows_to_mask)/2):]
    for j in rows_to_mask:
        del_masked = np.insert(del_masked, j, np.nan, axis=0)
        del_masked = np.insert(del_masked, j, np.nan, axis=1)
    del_masked = del_masked[0:target_length_cropped, 0:target_length_cropped]
    return wt_masked, del_masked, rows_to_mask

def make_del_preds(chrom, del_start, del_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19):

    chr_str = "{}-{}-{}".format(chrom, del_start, del_stop)
    print("Deletion: ", chr_str)
    center = (del_start+del_stop)//2
    region_start = center-half_patch_size
    region_stop = center+half_patch_size
    
    wt_mat = predict_wt(chrom, region_start, region_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19)
    sym_mat = predict_del(chrom, region_start, region_stop, del_start, del_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19)
    
    s_wt_masked, s_del_masked, s_del_idx = mask_mat(wt_mat, region_start, region_stop, 
                                              sym_mat, del_start, del_stop, target_length_cropped)
    return s_wt_masked, s_del_masked, s_del_idx,region_start,region_stop

def make_random_mut(chrom, seq_start, seq_end, mut_start, mut_end, fasta_open, half_patch_size=2**19):
    if seq_end-seq_start != (2*half_patch_size):
        to_add = 2*half_patch_size - (seq_end - seq_start)
        seq_end += to_add

    seq_mut_start = mut_start-seq_start
    seq_mut_end = mut_end-seq_start
    
    seq = fasta_open.fetch( chrom, seq_start, seq_end).upper()
    seq_1hot = one_hot_encode(seq)
    seq_mut_1hot = np.copy(seq_1hot)
    
    for mi in range(seq_mut_start, seq_mut_end):
        posi = 0
        poss = list(range(4))
        for ni in range(4):
            if seq_1hot[mi,ni] == 1:
                posi = ni
                seq_mut_1hot[mi,ni] =0
        poss.remove(posi)
        posr = random.choice(poss)
        seq_mut_1hot[mi,posr] =1
        
    return seq_mut_1hot

def predict_random_mut(region_chr, region_start, region_stop, mut_start, mut_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19):
    
    seq_1hot = make_random_mut(region_chr, region_start, region_stop, mut_start, mut_stop, fasta_open, half_patch_size=2**19)
    pred_targets = seqnn_model.predict(np.expand_dims(seq_1hot,0))    
    sym_mat = from_upper_triu(pred_targets[0,:,0],target_length_cropped,2)
    sym_mat_denan = interp_all_nans(sym_mat)

    return sym_mat_denan

def make_mut_preds(chrom, mut_start, mut_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19):

    chr_str = "{}-{}-{}".format(chrom, mut_start, mut_stop)
    print("Mutation: ", chr_str)
    center = (mut_start+mut_stop)//2
    region_start = center-half_patch_size
    region_stop = center+half_patch_size
    
    wt_mat = predict_wt(chrom, region_start, region_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19)
    mut_mat = predict_random_mut(chrom, region_start, region_stop, mut_start, mut_stop, fasta_open, seqnn_model, target_length_cropped, half_patch_size=2**19)
    return  wt_mat,mut_mat,region_start,region_stop
