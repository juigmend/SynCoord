#!/usr/bin/env python
# coding: utf-8

'''Data processing.'''

import importlib
from copy import deepcopy

from scipy import signal
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.fft import rfft, fftfreq

def print_ptdata_properties(ptdata):
    print(f'names:\n{vars(ptdata.names)}\n')
    print(f'labels:\n{vars(ptdata.labels)}\n')
    if ptdata.data:
        print('data:')
        for k in ptdata.data.keys():
            print('k = ',k,', shape =',ptdata.data[k].shape)
        print()
    if ptdata.vis: print(f'vis:\n{ptdata.vis}\n')
    if ptdata.other: print(f'other:\n{ptdata.other}\n')

class PtData:
    '''
    Data class.
    Attributes:
        names: empty field allowing for subfields for names used in commands like selection.
        labels: empty field allowing for subfields for labels used in visualisations.
        data: dictionary where each entry should be an N-D numpy array containig the data.
              Each entry is at the top level of the data hierarchy. Its order and index should
              directly correspond to topinfo.
        topinfo: information for each entry at the top level. See help(load_data).
        vis: dictionary to be used as keyword arguments for default visualisation parameters,
             when the data object is passsed to visualise_ptdata. 
        other: empty dictionary, for any other information.
    Methods:
        print(): prints attributes and shape of data
    Note:
        Use command vars to see content of subfields.
        Example:
            ptdata = PtData(topinfo)    # initialises data object
            ptdata.names.main = 'Niels' # put a subfield with a string in field 'names'.
            vars(ptdata.names)
            returns: {'main': 'Niels'}
    Args:
        topinfo: See help(load_data).
    '''
    def __init__(self,topinfo):
        class SubField: pass
        self.names = SubField()
        self.labels = SubField()
        self.data = {}
        self.topinfo = topinfo
        self.vis = {}
        self.other = {}

    def print(self):
        print_ptdata_properties(self)

def position_ptdata( preproc_data_folder, prop_path, annot_path=None,
                        max_n_files=None, print_durations=True, print_dim=True, **kwargs ):
    '''
    Args:
        preproc_data_folder: folder of preprocessed position data parquet files (e.g., r"~\preprocessed").
        prop_path: path for properties CSV file (e.g., r"~\properties.csv").
        Optional:
            annot_path: path for annotations CSV file (e.g., r"~\Pachelbel_Canon_in_D_String_Quartet.csv").
            max_n_files: number of files to extract from the beginning of annotations. None or Scalar.
            print_durations: print durations of data. True or False.
            **kwargs: passed through, see help(load_data).
    Returns:
        PtData object with position data (dictionary of mutlti dimensional numpy arrays).
    '''
    position_data, dim_names, dim_labels, dimel_labels, topinfo =\
        load_data( preproc_data_folder, prop_path, annot_path=annot_path, max_n_files=max_n_files,
                   print_durations=print_durations, print_dim=print_dim, **kwargs )

    position = PtData(topinfo)
    position.names.main = 'Position'
    position.names.dim = dim_names
    position.labels.main = 'Pos.'
    position.labels.dim = dim_labels
    position.labels.dimel = dimel_labels
    position.data = position_data
    position.vis['y_label'] = None
    return position

def select_data(ptdata,*args,**kwargs):
    '''
    Args:
        ptdata: PtData object
        Either of these:
            * Location for top-level and dimensions,
              as a list ordered as [top-level, dim 1, dim 2, ...].
            * Arbitrary number of keywords and values, separated or in a dictionary.
              Keywords are 'top' and dimension names.
            In both cases values are locations and may be 'all', int, or slice(start,end), separated
            or in a nested list with non-continguous values.
    Returns:
        sel: PtData object containing the selected data and topinfo.
    Examples:
        sel = select_data(ptdata, [1,slice(0,180),1,'all'])
        sel = select_data(ptdata, [1,[slice(0,180),slice(300,600)],1,'all']) # non-contiguous values for dim 1
        sel = select_data(ptdata, top = 2, point = 1, frame = 'all') # if frame is dim 1, keword value 'all' is redundant
        sel = select_data(ptdata, top = [0,1],  point = 1, frame = slice(0,600))
        sel = select_data(ptdata, {'top' : 0, 'point' : 1, 'frame' : slice(0,600)})
    '''
    try:
        if args[0] is None: args = ()
    except: pass
    if (not args) and (not kwargs): return ptdata
    warning_str = 'Input has to be either list, keywords, or dictionary.'
    if args and kwargs: raise Exception(warning_str)
    loc_sel = None
    name_sel = kwargs
    if args:
        for e in args:
            if isinstance(e,list): loc_sel = e
            if isinstance(e,dict):
                if name_sel: raise Exception(warning_str)
                name_sel = e
    sel = PtData(ptdata.topinfo)

    top_sel = None
    loc_dim = None
    if name_sel:
        top_names = ['top'] + list(ptdata.topinfo.columns)
        valid_names = list(ptdata.names.dim) + top_names
        other_keys = [k for k in name_sel.keys() if k not in valid_names]
        if other_keys: raise Exception(f'Invalid keys: {", ".join(other_keys)}')
        name_sel = {k:v for k,v in name_sel.items() if k in valid_names}
        if 'top' in name_sel:
            top_sel = name_sel.pop('top')
        loc_dim = [None for _ in range(len(ptdata.names.dim))]
        for i,dim_name in enumerate(ptdata.names.dim):
            if dim_name in name_sel:
                loc_dim[i] = name_sel[dim_name]
            else: loc_dim[i] = 'all'

    if loc_sel:
        top_sel = loc_sel[0]
        if len(loc_sel) > 1:
            loc_dim = loc_sel[1:]

    if (top_sel is not None) and (top_sel != 'all'):
        if isinstance(top_sel,slice):
            sel.topinfo = ptdata.topinfo.iloc[top_sel,:]
        elif isinstance(top_sel,list):
            sel.topinfo = sel.topinfo[sel.topinfo.index.isin(top_sel)]
        else:
            sel.topinfo = ptdata.topinfo.iloc[top_sel:top_sel+1,:]
    sel_tops = sel.topinfo.index

    if loc_dim:
        range_loc_dim = range(len(loc_dim))
        sel_dim = loc_dim.copy()
        for i,dim in enumerate(sel_dim):
            if isinstance(dim,list):
                l = []
                for v in dim:
                    if isinstance(v,slice): l.extend( list(range(v.start,v.stop+1)) )
                    else: l.append(v)
                sel_dim[i] = l

    sel_data_dict ={}
    for i_top in sel_tops: # so inhomogeneous dimensions (except last) result in error
        ndarr = ptdata.data[i_top]
        if loc_dim and ('all' in loc_dim):
            for i in range_loc_dim:
                if loc_dim[i] == 'all': sel_dim[i] = slice(0,ndarr.shape[i])
                else: sel_dim[i] = loc_dim[i]
        sel_data_dict[i_top] = ndarr[tuple(sel_dim)]

    sel.names.main = ptdata.names.main
    sel.labels.main = ptdata.labels.main
    sel.names.dim = []
    sel.labels.dim = []
    sel.labels.dimel = []
    sel.data = sel_data_dict
    sel.vis = deepcopy(ptdata.vis)
    sel.other = deepcopy(ptdata.other)

    # update information:
    d_keys = sel.data.keys()
    for i,dim in enumerate(sel_dim):
        if isinstance(dim,slice):
            sel.names.dim.append(ptdata.names.dim[i])
            sel.labels.dim.append(ptdata.labels.dim[i])
            if ptdata.labels.dimel[i]: sel.labels.dimel.append(ptdata.labels.dimel[i].copy())
            else: sel.labels.dimel.append(ptdata.labels.dimel[i])
        if isinstance(loc_dim[i],slice): # this block uses references
            to_slice = [] # data to update
            to_slice.append( sel.labels.dimel[i] ) # 1 top-level data
            if 'y_ticks' in sel.vis:
                to_slice.append( sel.vis['y_ticks'] ) # n top-level data
            if 'freq_bins' in sel.other:
                to_slice.append( sel.other['freq_bins'] ) # n top-level data
            for ts in to_slice:
                if isinstance(ts,list): # 1 top-level data
                    ts = ts[loc_dim[i]]
                elif isinstance(ts,dict): # n top-level data
                    for k in d_keys:
                        ts[k] = ts[k][loc_dim[i]]
    return sel

def smooth_ptdata(ptdata,**kwargs):
    '''
    Apply filter to ptdata, row-wise, to a dimension of N-D arrays (default is last dimension).
    Args:
        ptdata: PtData object. See help(PtData).
        filter_type: 'butter', 'mean'
        Options:
            axis: int or str, default = -1.
                  Note: axis is a dimension of the N-D array. 
                        The rightmost axis (-1) is the fastest changing.
            For 'butter': freq_response ('lowpass','highpass','bandpass'), cutoff_freq (float or list), 
                          order (int), bandwidth (float, only for 'bandpass')
            For 'mean': window_size (seconds; float or list)
    Returns:
        New PtData object.
    '''
    filter_type = kwargs.get('filter_type','butter')
    if filter_type is None: return ptdata
    freq_response = kwargs.get('freq_response','lowpass')
    cutoff_freq = kwargs.get('cutoff_freq',2)
    order = kwargs.get('order',3)
    window_size = kwargs.get('window_size',3)
    bandwidth = kwargs.get('bandwidth',None)
    axis = kwargs.get('axis',-1)
    if isinstance(axis,str):
        axis_lbl = axis
        axis = ptdata.names.dim.index(axis)
    elif isinstance(axis,int): axis_lbl = ptdata.names.dim[axis]
    else: raise Exception('axis should be either string or int')

    if filter_type == 'butter':
        if 'band' in freq_response:
            if bandwidth:
                bw_half = bandwidth/2
                bp_freq = []
                for f in cutoff_freq:
                    if isinstance(f,list):
                        raise Exception( 'When bandwidth is specified, the \
                                          elements of cutoff_freq should be scalars, not nested lists.' )
                    else:
                        bp_freq.append( [v if (v>0) else 0.001 for v in [f-bw_half,f+bw_half]] )
                cutoff_freq = bp_freq
            else:
                if (len(cutoff_freq) == 2) \
                and (not isinstance(cutoff_freq[0],list)) \
                and (not isinstance(cutoff_freq[1],list)): cutoff_freq = [cutoff_freq]
        multiband_param = cutoff_freq
        main_name =  f'Filtered ({freq_response})\n{ptdata.names.main}'
        other = dict(list(kwargs.items())[:4]+list(kwargs.items())[5:])
        def butter_(arr,b,a):
            return signal.filtfilt(b, a, arr)
    if filter_type == 'mean':
        multiband_param = window_size
        main_name =  f'Filtered (moving mean)\n{ptdata.names.main}'
        other = dict(list(kwargs.items())[4:])
        def mean_(arr,ws):
            return np.convolve( arr, np.ones(round(ws))/round(ws), mode='same')
    if isinstance(multiband_param,np.ndarray): pass
    elif isinstance(multiband_param,tuple): multiband_param = list(multiband_param)
    elif not isinstance(multiband_param,list): multiband_param = [multiband_param]
    n_mbp = len(multiband_param)

    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    vis_opt = ptdata.vis.copy()
    if n_mbp > 1:
        means = [ round(np.mean(f).item(),3) for f in multiband_param ]
        vis_opt['y_ticks'] = means
        if filter_type == 'butter':
            dim_names.insert(axis,'frequency')
            dim_labels.insert(axis,'freq.')
            dimel_labels.insert(axis, [f'{m}Hz' for m in means] )
        if filter_type == 'mean':
            dim_names.insert(axis,'window')
            dim_labels.insert(axis,'w.')
            dimel_labels.insert(axis, [f'w={m}s' for m in means] )
    dd_in = ptdata.data
    dd_out = {}
    for i_top in dd_in.keys():
        fps = ptdata.topinfo['fps'].iloc[i_top]
        out_matrix = []
        if (axis != 0) and (n_mbp > 1):
            s = list(dd_in[i_top].shape)
            idx = list(range(1,dd_in[i_top].ndim+1))
            idx_adj = [v-1 for v in idx[:axis]] + idx[axis:]
            transposed_axes = [idx[axis]-1] + idx_adj
        for mbp in multiband_param:
            if filter_type == 'butter':
                b, a = signal.butter(order, mbp, freq_response, fs=fps)
                arr_out = np.apply_along_axis(butter_, axis,  dd_in[i_top], b, a)
            elif filter_type == 'mean':
                arr_out = np.apply_along_axis(mean_, axis,  dd_in[i_top], int(mbp*fps))
            out_matrix.append(arr_out)
        if (axis != 0) and (n_mbp > 1):
            dd_out[i_top] = np.transpose( np.array(out_matrix) , np.argsort(transposed_axes) )
        else:
            dd_out[i_top] = np.squeeze(np.array(out_matrix))

    filtered = PtData(ptdata.topinfo)
    filtered.names.main = main_name
    filtered.names.dim = dim_names
    filtered.labels.main = ptdata.labels.main[:]
    filtered.labels.dim = dim_labels
    filtered.labels.dimel = dimel_labels
    filtered.data = dd_out
    filtered.vis = vis_opt
    filtered.other = other
    return filtered

def peaks_to_phase(ndarr,axis=-1):
    '''
    Generate ramps between signal peaks, with amplitude {-pi,pi}
    Args:
        N-D array
        Options:
            axis: int, default = -1
                  Note: axis is a dimension of the N-D array. The rightmost axis is the fastest changing.
    Returns:
        N-D array
    '''
    def pks2ph(sig):
        phi = np.zeros(len(sig))
        idx_pks = signal.find_peaks(sig)
        for i in range(len(idx_pks[0])-1):
            i_start = idx_pks[0][i]
            i_end = idx_pks[0][i+1]
            ramp_length = int(np.diff(idx_pks[0][i:i+2])[0])
            phi[i_start:i_end] = np.linspace( start = -np.pi, stop = np.pi, num = ramp_length )
        return phi
    return np.apply_along_axis(pks2ph,axis,ndarr)

def tder2D(ndarr_in,order=1):
    '''
    Differentiation per point. First order difference is euclidean distance
    among consecutive two-dimensional points [x,y]. Second order difference is simple difference.
    The operation is applied per consecutive pairs of rows in dimension -2 among dimension -1,
    and the output has the same shape as the input, except dimension -2 has half the size.
    Args:
        N-D array where length of dimension -2 is 2 (x,y) or a multiple of 2 (x1,y1,x2,y2,...)
        Optional:
            order: 1 (default) or 2
    Returns:
        N-D array
    '''
    if ndarr_in.ndim > 2: ndarr_out = np.empty(ndarr_in.shape)
    n_points = ndarr_in.shape[-2]//2
    dim_out = list(ndarr_in.shape)
    dim_out[-2] = n_points
    ndarr_out = np.empty(tuple(dim_out))
    diff_arr = np.empty(tuple(dim_out[-2:]))
    for idx in np.ndindex(ndarr_in.shape[:-2]):
        if ndarr_in.ndim == 2: ndarr_slc = ndarr_in
        else: ndarr_slc = np.squeeze(ndarr_in[idx,:,:])
        for i_point in range(n_points):
            i_row = i_point*2
            coldiff = np.diff(ndarr_slc[i_row:i_row+2,:])
            diff_arr[i_row,1:] = np.linalg.norm( coldiff, axis=0) # absolute 1st. order diff. (speed)
            diff_arr[i_row,0] = diff_arr[i_row,1]
            if order == 2:
                diff_arr[i_row,1:] = np.diff(diff_arr[i_row,:]) # 2nd. order diff. (acceleration)
                diff_arr[i_row,0] = diff_arr[i_row,1]
        if ndarr_in.ndim == 2:
            ndarr_out = diff_arr
        else:
            ndarr_out[idx,:,:] = diff_arr
    return np.squeeze(ndarr_out)

def apply_to_ptdata(ptdata, func,*args, **kwargs):
    '''
    Apply a function to every N-D array of the data dictionary in a PtData object.
    Note that ptdata.dim will be copied from the input and may not correspond to the output,
    except for these functions: tder2D, peaks_to_phase, kuramoto_r, power.
    Args:
        ptdata: PtData object, see help(PtData).
        func: a function to operate on each N-D array of the dictionary.
        Optional:
            axis: dimension to apply process. Default is -1.
            *args, **kwargs: input arguments and keyword-arguments to the function, respectively.
    Returns:
        New PtData object.
    '''
    axis = kwargs.get('axis',-1)
    args_list = list(args)
    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    vis = ptdata.vis.copy()
    fn = func.__name__
    dd_in = ptdata.data
    if fn == 'tder2D':
        del dim_names[axis-1]
        del dim_labels[axis-1]
        del dimel_labels[axis-1]
        main_name = 'Speed'
        main_label = '| $v$ |'
        if ('order' in kwargs) and (kwargs['order'] == 2):
            main_name = 'Absolute Acceleration'
            main_label = '| $a$ |'
    elif fn == 'peaks_to_phase':
        main_name = 'Peaks Phase'
        main_label = r'$\phi$'
        vis = {**vis, 'dlattr':'k0.2', 'vlattr':'r-3b'}
    elif fn == 'kuramoto_r':
        main_name = 'Kuramoto Order Parameter $r$'
        main_label = '$r$'
        del dim_names[axis-1]
        del dim_labels[axis-1]
        del dimel_labels[axis-1]
        vis = {**vis, 'dlattr':'1','vlattr':'r:2f'}
    elif fn == 'power':
        main_name = rf'{ptdata.names.main[:].capitalize()} $^{args_list[0]}$'
        main_label = rf'{ptdata.labels.main[:]}$^{args_list[0]}$'
    else:
        print('Warning: output "dim" field copied fromm input.')
        main_name = f'{fn.capitalize()}'

    dd_out = {}
    for k in dd_in.keys():
        if (fn not in ['tder2D','kuramoto_r']) or (dd_in[k].shape[axis-1] == 2):
            dd_out[k] = func(dd_in[k],*args,**kwargs)
        elif dd_in[k].shape[axis-1] > 2:
            arr_in = dd_in[k].copy()
            if arr_in.ndim > 2:
                try:
                    arr_in = np.swapaxes(arr_in, axis-2, axis-1)
                except:
                    arr_in = np.swapaxes(arr_in, axis-3, axis-1)
                if not dd_out:
                    dim_names[axis-1] = ptdata.names.dim[axis-1]
                    dim_labels[axis-1] = ptdata.labels.dim[axis-1]
                    dimel_labels[axis-1] = ptdata.labels.dimel[axis-1]
            s = list(arr_in.shape[:])
            s[axis-1] = 1
            arr_out = np.empty(tuple(s))
            for slc,_,midx in nd_iter(arr_in,lockdim=[axis-1,axis]):
                midx[axis-1] = 0
                midx_str = str(midx).replace("'","")
                exec('arr_out'+midx_str+'= func(slc,*args,**kwargs) ')
            dd_out[k] = np.squeeze( np.array(arr_out) )
        else:
            raise Exception('input should be at least 2D')

    processed = PtData(ptdata.topinfo)
    processed.names.main = main_name
    processed.names.dim = dim_names
    processed.labels.main = main_label
    processed.labels.dim = dim_labels
    processed.labels.dimel = dimel_labels
    processed.data = dd_out
    processed.vis = vis
    processed.other = ptdata.other.copy()
    return processed

def kuramoto_r(ndarr):
    '''
    Row-wise kuramoto order parameter r.
    Args:
        N-D array of phase angles,
        where dim = -2 is rows for points and dim = -1 is columns for observations.
        Singleton dimensions will be removed firstly.
    Returns:
        1-D array of Kuramoto order parameter r.
    '''
    if 1 in ndarr.shape: ndarr_sqz = np.squeeze(ndarr)
    else: ndarr_sqz = ndarr
    n_pts = ndarr_sqz.shape[-2]
    r_list = []
    for phi in ndarr_sqz.T:
        r_list.append( abs( sum([(np.e ** (1j * angle)) for angle in phi]) / n_pts ) )
    return np.array(r_list)

def isochronal_sections(data_list,idx_sections,axis=-1):
    '''
    Time-rescale 1-D data so that data fits into sections of the same size.
    The length of the resulting sections will be the length of the largest array index of sections.
    Args:
        data_list: a list of N-D arrays with the data.
        idx_sections: corresponding list of lists with the index of sections.
        axis: axis to apply the process.
              Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        isochr_data: list of arrays with the processed data.
        idx_isochr_sections: list with the index of isochronal sections.
    '''
    n_sections = np.inf
    for i in range(len(idx_sections)):
        l = len(idx_sections[i])
        if l < n_sections: n_sections = l
    max_length = 0
    for i in range(len(data_list)):
        if data_list[i].shape[axis] > max_length: max_length = data_list[i].shape[axis]
    length_section = round(max_length/n_sections)
    length_total = length_section * n_sections
    isochr_data = []

    def isochrsec_(arr_in,idx_sec,length_total,length_section,n_sections):
        arr_out = np.empty(length_total)
        i_r_start = 0 # index raw
        i_w_start = 0 # index time-rescaled by interpolation
        for i_section in range(n_sections):
            if i_section == n_sections-1: i_r_end = -1
            else: i_r_end = idx_sec[i_section]
            section_raw = arr_in[i_r_start : i_r_end ]
            i_r_start = i_r_end
            t_raw = np.linspace(0, len(section_raw)-1,  len(section_raw))
            interpol = CubicSpline(t_raw, section_raw)
            t_rescaled = np.linspace(0, len(section_raw)-1, num = int(length_section))
            i_w_end = i_w_start + length_section
            arr_out[ i_w_start : i_w_end ] = interpol(t_rescaled)
            i_w_start = i_w_end
        return arr_out

    for i_arr in range(len(data_list)):
        idx_sec = idx_sections[i_arr]
        arr_in = data_list[i_arr]
        isochrsec_result = np.apply_along_axis( isochrsec_, axis, arr_in, idx_sec, length_total, 
                                                length_section, n_sections )
        isochr_data.append( isochrsec_result )

    idx_isochr_sections = list(range(length_section,length_section*n_sections+1,length_section))
    return isochr_data, idx_isochr_sections

def isochrsec_ptdata(ptdata,axis=-1):
    '''
    Wrapper for isochronal_sections.
    Time-rescale data so that it fits into sections of the same length.
    The length of the resulting sections will be the length of the largest input index of sections.
    Args:
        ptdata: PtData object with topinfo containing column 'trimmed_sections_frames'.
                See help(load_data).
        Optional:
            axis: axis to apply the process.
                  Note: axis is a dimension of the N-D array.
                        The rightmost axis (-1) is the one that changes most frequently.
    Returns:
        New PtData object, with 'trimmed_sections_frames' modified.
    '''
    data_list = []
    for ndarr in ptdata.data.values():
        data_list.append(ndarr)
    idx_sections = ptdata.topinfo['trimmed_sections_frames'].values.tolist()
    isochr_data, idx_isochr_sections = isochronal_sections(data_list,idx_sections,axis)
    ddict = {}
    sec_list = []
    i_l = 0
    for k in ptdata.data.keys():
        ddict[k] = isochr_data[i_l]
        i_l += 1
        sec_list.append(idx_isochr_sections)
    new_topinfo = ptdata.topinfo.copy()
    new_topinfo['trimmed_sections_frames'] = sec_list
    isochrsec = PtData(new_topinfo)
    isochrsec.names.main = ptdata.names.main+'\n(time-rescaled isochronal sections)'
    isochrsec.names.dim = ptdata.names.dim.copy()
    isochrsec.labels.main = ptdata.labels.main
    isochrsec.labels.dim = ptdata.labels.dim.copy()
    isochrsec.labels.dimel = ptdata.labels.dimel.copy()
    isochrsec.data = ddict
    isochrsec.vis = ptdata.vis
    isochrsec.other = ptdata.other.copy()
    return isochrsec

def aggrsec_ptdata( ptdata, aggregate_axes=[-2,-1], sections_axis=1,
                    omit=None, function='mean' ):
    '''
    Aggregate sections.
    Args:
        ptdata: PtData object, see help(PtData).
                The field topinfo should have columns 'trimmed_sections_frames', 
                with lists having equally spaced indices. See help(isochrsec_ptdata).
        Optional:
            aggregate_axes: scalar or list indicating the axes to aggregate.
            sections_axis: the axes of the aggregated axes where the sections are taken from.
                  For example:
                      data.shape = (6,1,15,9000)
                      aggregate_axes = [-2,-1]:
                          shape of aggregated axes: (15,9000)
                      sections_axis = 1:
                          The sections are taken from the axis with length 9000 points.
                Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
            omit: int or list or sections to omit
            function: 'mean' or 'sum'
    Returns:
        New PtData object.
    '''
    i_sec_ax = aggregate_axes[sections_axis]
    if not isinstance(omit,list): omit = [omit]
    if not isinstance(aggregate_axes,list): aggregate_axes = [aggregate_axes]
    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in.keys():
        idx_isochr_sections = ptdata.topinfo.loc[k,'trimmed_sections_frames']
        n_sections = len(idx_isochr_sections)
        n_sec_str = str(n_sections)
        length_section = np.diff(idx_isochr_sections[:2]).item()
        iter_shape = list(dd_in[k].shape)
        shape_out = iter_shape.copy()
        loc_idx_iter = list(range(dd_in[k].ndim))
        idx_s_in = [None for _ in dd_in[k].shape]
        if dd_in[k].ndim < len(aggregate_axes):
            ag_ax = aggregate_axes[-dd_in[k].ndim:]
        else:
            ag_ax = aggregate_axes
        for i in ag_ax:
            del iter_shape[i]
            del loc_idx_iter[i]
            if i != i_sec_ax:
                idx_s_in[i] = ':'
        shape_out[i_sec_ax] = length_section
        dd_out[k] = np.zeros(tuple(shape_out))
        for idx_iter in np.ndindex(tuple(iter_shape)):
            for i_loc,i_idx in zip(loc_idx_iter,idx_iter):
                idx_s_in[i_loc] = i_idx
            idx_s_in[i_sec_ax] = "i_start:i_end"
            idx_s_in_str = str(idx_s_in).replace("'","")
            idx_s_out = idx_s_in.copy()
            idx_s_out[i_sec_ax] = ':'
            idx_s_out_str = str(idx_s_out).replace("'","")
            i_start = 0
            for i_sec in range(n_sections):
                i_sec_f = idx_isochr_sections[i_sec]
                i_end = i_sec_f
                if i_sec not in omit:
                    exec(''.join(['dd_out[k]',idx_s_out_str,' += dd_in[k]',idx_s_in_str]))
                i_start = i_end
            if function == 'mean':
                exec(''.join(['dd_out[k]',idx_s_out_str,' = dd_out[k]',idx_s_out_str,'/',n_sec_str]))

    main_name = ptdata.names.main
    isochr_lbl = '(time-rescaled isochronal sections)'
    if function == 'sum':
        function_lbl = 'added'
    elif function == 'mean':
        function_lbl = 'mean'
    aggr_lbl = f'({function_lbl} sections)'
    if isochr_lbl in main_name: main_name = main_name.replace(isochr_lbl,aggr_lbl)
    else: main_name = main_name + aggr_lbl
    aggrsec = PtData(ptdata.topinfo)
    aggrsec.names.main = main_name
    aggrsec.names.dim = ptdata.names.dim.copy()
    aggrsec.labels.main = ptdata.labels.main
    aggrsec.labels.dim = ptdata.labels.dim.copy()
    aggrsec.labels.dimel = ptdata.labels.dimel.copy()
    aggrsec.data = dd_out
    aggrsec.vis = ptdata.vis.copy()
    aggrsec.vis = {**aggrsec.vis, 'sections':False,'x_ticklabelling':'33.3%'}
    aggrsec.other = ptdata.other.copy()
    return aggrsec

def fourier( ndarr, window_length, fps=None, output='spectrum', window_shape=None,
             mode='same', first_fbin=1, axis=-1 ):
    '''
    Wrapper for scipy.fft.rfft
    Fast Fourier transform for a signal of real numbers.
    Args:
        ndarr: N-D array
        window_length: length of the FFT window vector, in seconds if the fps parameter is given.
        Options:
            fps
            output: 'spectrum' or 'phase'(radians).
            window_shape: main_name of the window shape (eg.'hann'). See help(scipy.signal.windows) or
                          https://docs.scipy.org/doc/scipy/reference/signal.windows.html
            mode: 'same' (zero-padded, same size of input) or 'valid' (only FFT result).
            first_fbin: Remove frequency bins under this number. Default = 1 (removes DC offset).
            axis: int, default = -1 (last dimension of the N-D array).
                  Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        N-D array, whose two last two dimensions are the result of this function.
    '''
    if fps: window_length = round(window_length * fps)
    fft_window = 1
    if window_shape:
        windows_module = importlib.import_module('scipy.signal.windows')
        window_func = eval('windows_module.'+window_shape)
        fft_window = window_func(window_length)
    zpad = mode=='same'

    def fourier_( sig, fft_window, window_length, zpad, first_fbin, output ):
        fft_result = []
        for i_window in range(len(sig)-window_length):
            this_window = sig[ i_window : i_window+window_length] * fft_window
            this_spectrum = rfft(this_window)[first_fbin:]
            if output == 'spectrum': fft_result.append(this_spectrum)
            elif output == 'phase': fft_result.append( np.angle(this_spectrum) )
        fft_result = np.array(fft_result).T
        if zpad:
            dif = len(sig) - fft_result.shape[1]
            margin = np.floor(dif/2).astype(int)
            fft_result = np.pad( fft_result, ( (0,0),(margin, margin + int(dif%2) )) )
        return fft_result
    return np.apply_along_axis( fourier_, axis, ndarr, fft_window, window_length, zpad,
                                first_fbin, output )

def fourier_ptdata(ptdata, window_length, **kwargs):
    '''
    Wrapper for fourier. See help(fourier).
    Args:
        ptdata: PtData object, see help(PtData).
        window_length: length of the FFT window in seconds unless optional parameter fps = None.
        Optional:
            **kwargs: input parameters to the fourier function. See help(fourier).
    Returns:
        New PtData object.
    '''
    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    dd_in = ptdata.data

    if ('output' in kwargs) and (kwargs['output'] == 'phase'): 
        main_name = 'Phase'
        dim_names.insert(-1,'frequency')
        main_label = r'$\phi$'
        dim_labels.insert(-1,'freq.')
    else:
        main_name = 'Frequency Spectrum'
        dim_names.insert(-1,'amplitude')
        main_label = 'Spectrum'
        dim_labels.insert(-1,'amp.')

    first_fbin = kwargs.get('first_fbin',1)
    freq_bins = {}
    freq_bins_labels = {}
    wl = window_length

    dd_out = {}
    for k in dd_in:
        fps = ptdata.topinfo.loc[k,'fps']
        if 'fps' not in kwargs: wl = round(window_length * fps)
        dd_out[k] = fourier(dd_in[k], wl,**kwargs)
        freq_bins[k] = np.abs(( fftfreq(wl)*fps )[first_fbin:np.floor(wl/2 + 1).astype(int)])
        freq_bins_rounded = np.round(freq_bins[k],2)
        rdif = abs(np.mean(freq_bins_rounded-np.round(freq_bins_rounded)))
        if rdif < 0.01: freq_bins_rounded = np.round(freq_bins_rounded,0).astype(int)
        freq_bins_labels[k] = [str(f)+' Hz' for f in freq_bins_rounded]
    dimel_labels.insert(-1,freq_bins_labels)
    other = {'freq_bins':freq_bins}

    fft_result = PtData(ptdata.topinfo)
    fft_result.names.main = main_name
    fft_result.names.dim = dim_names
    fft_result.labels.main = main_label
    fft_result.labels.dim = dim_labels
    fft_result.labels.dimel = dimel_labels
    fft_result.data = dd_out
    fft_result.vis = {'dlattr':'k0.8'}
    fft_result.other = other
    return fft_result

def slwin(arrs, window_length, func, mode='same', **kwargs):
    '''
    Apply a function to a sliding window over the last dimension (-1) of one or two numpy arrays.
    Args:
        arrs: 1-D or N-D array or a list with two of such arrays having the same dimensions.
        window_length: length of the window vector.
        func: function to apply, with one or two required inputs (consistent with arrs).
        Optional:
            mode: 'same' (zero-padded, same size of input) or 'valid' (only func result).
            **kwargs = keyword arguments to be passed to func.
    Returns:
        Array whose dimensions depend on func.
    '''
    if isinstance(arrs,list): len_arr = arrs[0].shape[-1]
    else: len_arr = arrs.shape[-1]
    slwin_result = []
    m_bit = 0
    for i_window in range(len_arr-window_length):
        if isinstance(arrs,np.ndarray):
            this_window = arrs[ ..., i_window : i_window + window_length ]
            this_result = func(this_window,**kwargs)
        elif isinstance(arrs,list):
            this_window_i = arrs[0][ ..., i_window : i_window + window_length ]
            this_window_j = arrs[1][ ..., i_window : i_window + window_length ]
            this_result = func(this_window_i,this_window_j,**kwargs)
        slwin_result.append(this_result)
    slwin_result = np.array(slwin_result).T
    if mode == 'same':
        dif = len_arr - slwin_result.shape[1]
        margin = np.floor(dif/2).astype(int)
        slwin_result = np.pad( slwin_result, ( (0,0),(margin, margin + int(dif%2) )) )
    return slwin_result

def pfndarr(ndarr, func, pairs_axis, fixed_axes=-1, verbose=True, **kwargs):
    '''
        Apply a function to pairs of dimensions of an N-D array.
        Args:
            ndarr: N-D array.
            func: function to apply, whose first argument is a list with each N-D array of the pair.
            pairs_axis: axis to run the pairwise process.
            Optional:
                fixed_axes: axis (int) or axes (list) that are the input to func. Default is last axis *.
                verbose: Display progress.
                **kwargs = optional arguments and keyword arguments to be passed to func.
        Returns:
            N-D array. The length of the pairs dimension originanlly of length N, is ((N*N)-N)/2.
            List of pairs.
        * axes = dimensions of the N-D array, where the rightmost axis is the fastest changing.
    '''
    shape_in = list(ndarr.shape)
    iter_shape = shape_in.copy()
    n_pair_el = iter_shape.pop(pairs_axis)
    n_pairs = (n_pair_el**2 - n_pair_el)//2
    loc_idx_iter = list(range(len(shape_in)))
    del loc_idx_iter[pairs_axis]
    if not isinstance(fixed_axes,list): fixed_axes = [fixed_axes]
    idx_shape_o = [None for _ in shape_in]
    for i in fixed_axes:
        del iter_shape[i]
        del loc_idx_iter[i]
        idx_shape_o[i] = ':'
    idx_shape_i = idx_shape_o.copy()
    idx_shape_j = idx_shape_o.copy()
    shape_out = shape_in.copy()
    shape_out[pairs_axis] = (shape_in[pairs_axis]**2 - shape_in[pairs_axis])//2
    ouput_ndarr = np.empty(tuple(shape_out))
    len_pairs_axis = ndarr.shape[pairs_axis]
    i_pair = 0
    pairs_idx = []
    for i in range(len_pairs_axis):
        idx_shape_i[pairs_axis] = i
        for j in range(i+1, len_pairs_axis):
            idx_shape_j[pairs_axis] = j
            idx_shape_o[pairs_axis] = i_pair
            if verbose: print(f'pair {i_pair+1} of {n_pairs}')
            pairs_idx.append([i,j])
            i_pair += 1
            for idx_iter in np.ndindex(tuple(iter_shape)):
                for i_loc,i_idx in zip(loc_idx_iter,idx_iter):
                    idx_shape_i[i_loc] = i_idx
                    idx_shape_j[i_loc] = i_idx
                    idx_shape_o[i_loc] = i_idx
                idx_shape_i_str = str(idx_shape_i).replace("'","")
                idx_shape_j_str = str(idx_shape_j).replace("'","")
                idx_shape_o_str = str(idx_shape_o).replace("'","")
                slice_i = eval('ndarr'+idx_shape_i_str)
                slice_j = eval('ndarr'+idx_shape_j_str)
                exec('ouput_ndarr'+idx_shape_o_str+' = func([slice_i,slice_j],**kwargs)')
    return ouput_ndarr, pairs_idx

def phasediff(phi_1,phi_2):
    '''
    Args:
        phi_1, phi_2: scalars of vectors that are phase angles.
    Returns:
        Phase difference.
    '''
    return np.arctan2( np.cos(phi_1) * np.sin(phi_2) - np.sin(phi_1) * np.cos(phi_2),
                       np.cos(phi_1) * np.cos(phi_2) + np.sin(phi_1) * np.sin(phi_2) )

def plv(a1,a2,axis=0):
    '''
    Args:
        a1, a2: phase angles.
        Optional:
            axis
            Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        Phase-locking value.
    '''
    diff_complex = np.exp(complex(0,1)*(a1-a2))
    plv_result = np.abs(np.sum(diff_complex,axis=axis))/len(diff_complex)
    return plv_result

def wplv(arrs, window_length=None, mode='same', axis=1):
    '''
    Phase-locking value on a sliding window over two numpy arrays.
    Args:
        arrs: 1-D array or a list with two 1-D arrays with the same length.
        window_length: length of the window vector.
        Optional:
            mode: 'same' (zero-padded, same size of input) or 'valid' (only func result).
            axis to apply plv
                Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        Array whose dimensions depend on func.
    '''
    return slwin( arrs, window_length, plv, mode=mode, axis=axis)

def wplv_ptdata( ptdata, window_duration, pairs_axis=0, fixed_axes=[-2,-1],
                 plv_axis=1, mode='same', verbose=False ):
    '''
    Pairwise windowed Phase-Locking Value.
    Args:
        ptdata: PtData object, see help(PtData).
        window_duration: window length in seconds
        Optional:
            pairs_axis: axis to form the pairs.
            fixed_axes: axis (int) or axes (list) that are passed to the windowed PLV function.
            plv_axis: axis from the fixed axes, that the windowed PLV function will be performed upon.
                      For example:
                          data.shape = (4,1,15,9000)
                          pairs_axis = 0:
                              6 pairs to be formed: ([0,1],[0,2],[0,3],[1,2],[1,3],[2,3])
                          fixed_axes = [-2,-1]:
                              Passed to the windowed PLV function: data.shape = (15,9000)
                          plv_axis = -1:
                              The PLV function will be applied across the 9000 points of
                              each of the 15 vectors.
                Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
            mode: 'same' (zero-padded, same size as input) or 'valid' (result of windowed process)
            verbose: Display progress.
    Returns:
        New PtData object.
    '''
    dd_in = ptdata.data
    dd_out = {}
    c = 1
    for k in dd_in.keys():
        if verbose:
            print(f'processing array {k} ({c} of {len(ptdata.data.keys())})')
            c+=1
        window_length = round(window_duration * ptdata.topinfo.loc[k,'fps'])
        dd_out[k], pairs_idx = pfndarr( dd_in[k], wplv, pairs_axis, window_length=window_length,
                                        fixed_axes=fixed_axes, mode=mode, axis=plv_axis,
                                        verbose=verbose )
    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    if mode == 'valid':
        i_wlbl = fixed_axes[plv_axis]
        if i_wlbl < 0: i_wlbl = len(dim_names) + i_wlbl
        dim_names[i_wlbl] = dim_labels[i_wlbl] = 'window'
    if isinstance(fixed_axes,list): i_nlbl = fixed_axes[0]
    else: i_nlbl = fixed_axes
    if i_nlbl < 0: i_nlbl = len(dim_names) + i_nlbl - 1
    if (i_nlbl != pairs_axis) and (i_nlbl >= 0):
        dim_names[i_nlbl] = dim_labels[i_nlbl] = 'PLV'
    dim_names[pairs_axis] = 'pair'
    dim_labels[pairs_axis] = 'pairs'
    k = list(dd_in.keys())[0]
    n_pair_el = dd_in[k].shape[pairs_axis]
    n_pairs = (n_pair_el**2 - n_pair_el)//2
    dimel_labels[pairs_axis] = ['pair '+str(p) for p in pairs_idx]

    winplv = PtData(ptdata.topinfo)
    winplv.names.main = 'Pairwise Windowed Phase-Locking Value'
    winplv.names.dim = dim_names
    winplv.labels.main = 'PLV'
    winplv.labels.dim = dim_labels
    winplv.labels.dimel = dimel_labels
    winplv.data = dd_out
    winplv.vis = {'groupby':fixed_axes[0], 'vistype':'imshow', 'vlattr':'r:3f'}
    if 'freq_bins' in ptdata.other:
        winplv.vis['y_ticks'] = ptdata.other['freq_bins'].copy()
    winplv.other = ptdata.other.copy()
    return winplv

def aggrax_ptdata( ptdata, axis=0, function='mean' ):
    '''
    Wrapper for np.sum or np.mean
    Args:
        ptdata: PtData object, see help(PtData).
        Optional:
            function: 'sum' or 'mean'
            axis to run the operation
                Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        New PtData object.
    '''
    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in.keys():
        if function == 'sum': dd_out[k] = np.sum(dd_in[k],axis=axis)
        elif function == 'mean': dd_out[k] = np.mean(dd_in[k],axis=axis)
    main_name = ptdata.names.main
    if function == 'sum':
        function_lbl = 'added'
    elif function == 'mean':
        function_lbl = 'mean'
    aggr_lbl = f'{function_lbl} dim. "{ptdata.names.dim[axis]}"'
    if main_name[-1] == ')': main_name = ''.join([main_name[:-1],', ',aggr_lbl,')'])
    else: main_name = ''.join([main_name,'\n(',aggr_lbl,')'])
    dim_names = ptdata.names.dim.copy()
    del dim_names[axis]
    dim_labels = ptdata.labels.dim.copy()
    del dim_labels[axis]
    aggrax = PtData(ptdata.topinfo)
    aggrax.names.main = main_name
    aggrax.names.dim = dim_names
    aggrax.labels.main = ptdata.labels.main
    aggrax.labels.dim = dim_labels
    aggrax.labels.dimel = dim_labels
    aggrax.data = dd_out
    aggrax.vis = ptdata.vis.copy()
    aggrax.vis = {**aggrax.vis, 'groupby':axis, 'sections':False,'x_ticklabelling':'33.3%'}
    aggrax.other = ptdata.other.copy()
    return aggrax
