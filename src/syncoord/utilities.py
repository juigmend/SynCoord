#!/usr/bin/env python
# coding: utf-8

'''Miscellaneous functionality.'''

import numpy as np

def halt(*arg):
    if not arg: arg = ('*** halt ***',)
    raise Exception(arg[0])

def frames_to_minsec_str(frames,fps):
    '''
    Convert frames (int) into a string with format "minutes:seconds".
    '''
    mins, sec = divmod(round(frames/fps), 60)
    if sec == 0: lbl = f'{mins}:00'
    elif sec < 10: lbl = f'{mins}:0{sec}'
    else: lbl = f'{mins}:{sec}'
    return lbl

def minsec_str_to_frames(ts_str_in,fps):
    '''
    Args:
        ts_str_in: comma-separated string of times as seconds, minutes:seconds, or minutes:seconds.decimals
        fps
    Returns:
        list of int: frames
    '''
    if not ts_str_in[-1].isnumeric():
        ts_str_in = ts_str_in[:-1]
    ts_str_in = ts_str_in.replace(' ','')
    ts_str_lst = ts_str_in.split(',')
    ts_f_lst = []
    for ts_str in ts_str_lst:
        if ':' not in ts_str:
            s = float(ts_str)
        else:
            s = sum(x * float(t) for x, t in zip([60, 1], ts_str.split(":")))
        ts_f_lst.append( round(s*fps) )
    return ts_f_lst

def trim_sections_to_frames(topinfo):
    '''
    Args:
        DataFrame with columns 'Start', 'Sections' and 'fps'.
        help(minsec_str_to_frames) for the format of cells in column 'Sections'.
    Returns:
        List with processed values.
    '''
    ts_f = []
    for i in topinfo.index:
        s_f = minsec_str_to_frames( topinfo['Sections'].iloc[i] , topinfo['fps'].iloc[i] )
        offset_s = topinfo['Start'].iloc[i]
        if np.isnan(offset_s): offset_s = 0
        offset_f = offset_s * topinfo['fps'].iloc[i]
        ts_f.append( [ round(f - offset_f) for f in s_f ] )
    return ts_f

def nd_iter(d,lockdim=None):
    '''
    Generator that iterates over dimensions of an N-D array.
    Args:
        d: N-D array (C-style, row major), or tuple with the shape of an array.
        Optional:
            lockdim: dimension (int) or dimensions (list) to not iterate over.
    Returns:
        arr_out: slice of d corresponding to the iteration, only if d is np.ndarray. 
                 If d is a tuple this is not returned.
        i_chdim: (list) index of changing dimensions.
        multi_idx: (list) multi-index of the current slice. If lockdim is not None, 
                   elements at lockdim are replaced with ':'.
    '''
    if isinstance(d,tuple):
        s = list(d)
        disarray = False
    elif isinstance(d,np.ndarray):
        s = list(d.shape)
        disarray = True
    if lockdim is not None:
        if not isinstance(lockdim,list):
            lockdim = [lockdim]
        for i,v in enumerate(lockdim):
            if v<0:
                lockdim[i] = len(s)+v
        for i in sorted(lockdim, reverse=True):
            del s[i]
    n_dim = len(s)
    i_end = n_dim-1
    dyn_idx = [0 for _ in range(n_dim)]
    multi_idx_prev = dyn_idx.copy()
    if lockdim is not None:
        for i in lockdim: multi_idx_prev.insert(i,':')
    for ii in range(len(multi_idx_prev)-1,-1,-1):
        if multi_idx_prev[ii] != ':':
            multi_idx_prev[ii] = None
            break

    while True:

        multi_idx = dyn_idx.copy()
        if lockdim is not None:
            for i in lockdim: multi_idx.insert(i,':')
        i_chdim = []
        for i,t in enumerate(zip(multi_idx_prev,multi_idx)):
            if t[0] != t[1]:
                i_chdim.append(i)
        multi_idx_prev = multi_idx.copy()
        if disarray:
            multi_idx_str = str(multi_idx).replace("'","")
            arr_out = eval('d'+multi_idx_str)
            yield arr_out, i_chdim, multi_idx
        else:
            yield i_chdim, multi_idx

        if sum([abs(s[i]-1-dyn_idx[i]) for i in range(n_dim) ]) == 0: break
        if dyn_idx[i_end] == s[i_end]-1:
            dyn_idx[i_end] = 0
            i = i_end - 1
            while True:
                dyn_idx[i] += 1
                if dyn_idx[i] == s[i]:
                    dyn_idx[i] = 0
                    i -= 1
                else: break
        else: dyn_idx[i_end] += 1
