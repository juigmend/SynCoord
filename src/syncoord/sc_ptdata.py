'''Data class PtData and functions that have ptdata objects as input.'''

from copy import deepcopy

import numpy as np
from scipy import signal
from scipy.fft import fftfreq
import matplotlib.pyplot as plt

from . import sc_ndarr, sc_utils

class PtData:
    '''
    Data class.
    Attributes:
        names: empty field allowing for subfields for names used in commands like selection.
        labels: empty field allowing for subfields for labels used in visualisations.
        data: dictionary where each entry should be an N-D numpy array containig the data.
              Each entry is at the top level of the data hierarchy. Its order and index should
              directly correspond to topinfo.
        topinfo: information for each entry at the top level. See help(sc_utils.load_data).
        vis: dictionary to be used as keyword arguments for default visualisation parameters,
             when the data object is passsed to sc_ptdata.visualise.
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
        topinfo: See help(sc_utils.load_data).
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
        print(f'names:\n{vars(self.names)}\n')
        print(f'labels:\n{vars(self.labels)}\n')
        if self.data:
            print('data:')
            for k in self.data.keys():
                print('k = ',k,', shape =',self.data[k].shape)
            print()
        if self.vis: print(f'vis:\n{self.vis}\n')
        if self.other: print(f'other:\n{self.other}\n')

def position( preproc_data_folder, prop_path, annot_path=None, max_n_files=None,
              print_durations=True, **kwargs ):
    '''
    Args:
        preproc_data_folder: folder of preprocessed position data parquet files (e.g., r"~\preprocessed").
        prop_path: path for properties CSV file (e.g., r"~\properties.csv").
        Optional:
            annot_path: path for annotations CSV file (e.g., r"~\Pachelbel_Canon_in_D_String_Quartet.csv").
            max_n_files: number of files to extract from the beginning of annotations. None or Scalar.
            print_durations: print durations of data. True or False.
            **kwargs: passed through, see help(sc_utils.load_data).
    Returns:
        PtData object with position data (dictionary of mutlti dimensional numpy arrays).
    '''
    position_data, dim_names, dim_labels, dimel_labels, topinfo =\
        sc_utils.load_data( preproc_data_folder, prop_path, annot_path=annot_path, max_n_files=max_n_files,
                   print_durations=print_durations, **kwargs )

    pos = PtData(topinfo)
    pos.names.main = 'Position'
    pos.names.dim = dim_names
    pos.labels.main = 'Pos.'
    pos.labels.dim = dim_labels
    pos.labels.dimel = dimel_labels
    pos.data = position_data
    pos.vis['y_label'] = None
    return pos

def select(ptdata,*args,**kwargs):
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
            if ptdata.labels.dimel[i]:
                if isinstance(ptdata.labels.dimel[i],str):
                    sel.labels.dimel.append(ptdata.labels.dimel[i])
                else: sel.labels.dimel.append(ptdata.labels.dimel[i].copy())
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

    if (ndarr.shape[-2] > 1) and (isinstance(loc_dim[-2],int)):
        if isinstance(ptdata.labels.dimel[-2],list):
            subtitle = ptdata.labels.dimel[-2][loc_dim[-2]]
        elif isinstance(ptdata.labels.dimel[-2],dict):
            if len(ptdata.labels.dimel[-2]) == 1:
                k = list(ptdata.labels.dimel[-2].keys())
                subtitle = ptdata.labels.dimel[-2][k[0]][loc_dim[-2]]
            else:
                new_dimel_labels = [v for v in ptdata.labels.dimel[-2][loc_dim[-2]].values()]
                if len(set(new_dimel_labels)) == 1:
                    subtitle = new_dimel_labels[0]
                else:
                    subtitle = ''
        sel.names.main = sel.names.main + f'\n{subtitle}'
    return sel

def smooth(ptdata,**kwargs):
    '''
    Apply filter to ptdata, row-wise, to a dimension of N-D arrays (default is last dimension).
    Args:
        ptdata: PtData object. See help(sc_ptdata.PtData).
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

def apply(ptdata, func,*args, **kwargs):
    '''
    Apply a function to every N-D array of the data dictionary in a PtData object.
    Note that ptdata.dim will be copied from the input and may not correspond to the output,
    except for these functions: tder2D, peaks_to_phase, kuramoto_r, power.
    Args:
        ptdata: PtData object, see help(sc_ptdata.PtData).
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
            for slc,_,midx in sc_ndarr.iter(arr_in,lockdim=[axis-1,axis]):
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

def isochrsec(ptdata,axis=-1):
    '''
    Wrapper for isochronal_sections.
    Time-rescale data so that it fits into sections of the same length.
    The length of the resulting sections will be the length of the largest input index of sections.
    Args:
        ptdata: PtData object with topinfo containing column 'trimmed_sections_frames'.
                See help(sc_utils.load_data).
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
    isochr_data, idx_isochr_sections = sc_ndarr.isochronal_sections(data_list,idx_sections,axis)
    ddict = {}
    sec_list = []
    i_l = 0
    for k in ptdata.data.keys():
        ddict[k] = isochr_data[i_l]
        i_l += 1
        sec_list.append(idx_isochr_sections)
    new_topinfo = ptdata.topinfo.copy()
    new_topinfo['trimmed_sections_frames'] = sec_list
    isec = PtData(new_topinfo)
    isec.names.main = ptdata.names.main+'\n(time-rescaled isochronal sections)'
    isec.names.dim = ptdata.names.dim.copy()
    isec.labels.main = ptdata.labels.main
    isec.labels.dim = ptdata.labels.dim.copy()
    isec.labels.dimel = ptdata.labels.dimel.copy()
    isec.data = ddict
    isec.vis = ptdata.vis
    isec.other = ptdata.other.copy()
    return isec

def aggrsec( ptdata, aggregate_axes=[-2,-1], sections_axis=1,
                    omit=None, function='mean' ):
    '''
    Aggregate sections.
    Args:
        ptdata: PtData object, see help(sc_ptdata.PtData).
                The field topinfo should have columns 'trimmed_sections_frames',
                with lists having equally spaced indices. See help(sc_ptdata.isochrsec).
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
    asec = PtData(ptdata.topinfo)
    asec.names.main = main_name
    asec.names.dim = ptdata.names.dim.copy()
    asec.labels.main = ptdata.labels.main
    asec.labels.dim = ptdata.labels.dim.copy()
    asec.labels.dimel = ptdata.labels.dimel.copy()
    asec.data = dd_out
    asec.vis = ptdata.vis.copy()
    asec.vis = {**asec.vis, 'sections':False,'x_ticklabelling':'33.3%'}
    asec.other = ptdata.other.copy()
    return asec

def fourier(ptdata, window_length, **kwargs):
    '''
    Wrapper for sc_ndarr.fourier_transform.
    Args:
        ptdata: PtData object, see help(sc_ptdata.PtData).
        window_length: length of the FFT window in seconds unless optional parameter fps = None.
        Optional:
            **kwargs: input parameters to the fourier_transform function. See help(sc_ndarr.fourier_transform).
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
        dd_out[k] = sc_ndarr.fourier_transform(dd_in[k], wl,**kwargs)
        freq_bins[k] = np.abs(( fftfreq(wl)*fps )[first_fbin:np.floor(wl/2 + 1).astype(int)])
        freq_bins_rounded = np.round(freq_bins[k],2)
        rdif = abs(np.mean(freq_bins_rounded-np.round(freq_bins_rounded)))
        if rdif < 0.01: freq_bins_rounded = np.round(freq_bins_rounded,0).astype(int)
        freq_bins_labels[k] = [f'bin {i}: {f} Hz' for i,f in enumerate(freq_bins_rounded)]
    dimel_labels.insert(-1,freq_bins_labels)
    other = {'freq_bins':freq_bins}

    fft_result = PtData(ptdata.topinfo)
    fft_result.names.main = main_name
    fft_result.names.dim = dim_names
    fft_result.labels.main = main_label
    fft_result.labels.dim = dim_labels
    fft_result.labels.dimel = dimel_labels
    fft_result.data = dd_out
    fft_result.vis = {'dlattr':'k0.8','vlattr':'r:2f'}
    fft_result.other = other
    return fft_result

def winplv( ptdata, window_duration, pairs_axis=0, fixed_axes=None,
            plv_axis=-1, mode='same', verbose=False ):
    '''
    Pairwise windowed Phase-Locking Value.
    Args:
        ptdata: PtData object, see help(sc_ptdata.PtData).
                N-D arrays should have at least 2 dimensions.
        window_duration: window length in seconds
        Optional:
            pairs_axis: axis to form the pairs.
            fixed_axes: axis (int) or axes (list) that are passed to the windowed PLV function.
                        Defaults: [-2,-1] if N-D array dimensions are 3 or more; -1 if 2 dimensions.
            plv_axis: axis to perform the windowed PLV function.
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
        if fixed_axes is None:
            if dd_in[k].ndim > 2:
                fixed_axes = [-2,-1]
            elif dd_in[k].ndim == 2:
                fixed_axes = -1
            if dd_in[k].ndim < 2:
                raise Exception('number of dimensions in data arrays should be at least 2')
        dd_out[k], pairs_idx = sc_ndarr.apply_to_pairs( dd_in[k], sc_ndarr.windowed_plv,
                                                        pairs_axis, window_length=window_length,
                                                        fixed_axes=fixed_axes, mode=mode, axis=plv_axis,
                                                        verbose=verbose )
    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    if mode == 'valid':
        i_wlbl = plv_axis
        if i_wlbl < 0: i_wlbl = len(dim_names) + i_wlbl
        dim_names[i_wlbl] = dim_labels[i_wlbl] = 'window'
    if isinstance(fixed_axes,list): i_nlbl = fixed_axes[0]
    else: i_nlbl = fixed_axes
    groupby = i_nlbl
    if i_nlbl < 0: i_nlbl = len(dim_names) + i_nlbl - 1
    if (i_nlbl != pairs_axis) and (i_nlbl >= 0):
        dim_names[i_nlbl] = dim_labels[i_nlbl] = 'PLV'
    dim_names[pairs_axis] = 'pair'
    dim_labels[pairs_axis] = 'pairs'
    k = list(dd_in.keys())[0]
    n_pair_el = dd_in[k].shape[pairs_axis]
    n_pairs = (n_pair_el**2 - n_pair_el)//2
    dimel_labels[pairs_axis] = ['pair '+str(p) for p in pairs_idx]
    if groupby == -1: vistype = 'line'
    else: vistype = 'imshow'

    wplv = PtData(ptdata.topinfo)
    wplv.names.main = 'Pairwise Windowed Phase-Locking Value'
    wplv.names.dim = dim_names
    wplv.labels.main = 'PLV'
    wplv.labels.dim = dim_labels
    wplv.labels.dimel = dimel_labels
    wplv.data = dd_out
    wplv.vis = {'groupby':groupby, 'vistype':vistype, 'vlattr':'r:3f'}
    if 'freq_bins' in ptdata.other:
        wplv.vis['y_ticks'] = ptdata.other['freq_bins'].copy()
    wplv.other = ptdata.other.copy()
    return wplv

def aggrax( ptdata, axis=0, function='mean' ):
    '''
    Wrapper for np.sum or np.mean
    Args:
        ptdata: PtData object, see help(sc_ptdata.PtData).
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
    agg = PtData(ptdata.topinfo)
    agg.names.main = main_name
    agg.names.dim = dim_names
    agg.labels.main = ptdata.labels.main
    agg.labels.dim = dim_labels
    agg.labels.dimel = dim_labels
    agg.data = dd_out
    agg.vis = ptdata.vis.copy()
    agg.vis = {**agg.vis, 'groupby':axis, 'sections':True,'x_ticklabelling':'33.3%'}
    agg.other = ptdata.other.copy()
    return agg

def visualise( ptdata, **kwargs ):
    '''
    Visualise data of a PtData object. Normally the object should contain information to generate the
    visualisation, mostly in the 'vis' field. Default settings may be changed with optional arguments.
    Args:
        ptdata: PtData object, see help(sc_ptdata.PtData).
        Optional:
            vistype: 'line', 'spectrogram', or 'imshow'
            groupby: int, str, or list, indicating N-D array's dimensions to group.
                     'default' = use defaults: line = 0, spectrogram = -2
            vscale: float, vertical scaling.
            dlattr: string, data lines' attributes colour, style, and width (e.g. 'k-0.6')
            sections: display vertical lines for sections. True or False.
            vlattr: vertical lines' attributes. String with one character for
                    colour, style, width, f (full vertical) or b (bits at the top and bottom).
                    For example: 'r:2f' means red, dotted, width=2, full vertical line.
            snum_hvoff: list with horizontal and vertical offset factor for section numbers.
            y_label: label for vertical axis. 'default' uses ptdata.labels.main
            y_ticks: labels for vertical axis ticks, useful only when vistype = 'imshow'
            x_ticklabelling: labelling of horizontal axis;
                             'default´'
                             's' = 'time (seconds)',
                             '25%' = xticks as percentage
                             'dim x' = use ptdata.labels.dim[x], or None.
            figtitle: figure title. If None, ptdata.name.main will be used.
            axes: dimensions to visualise. One axis for 'line' and'spectrogram', two axes for 'imshow'.
            sel_list: selection to display with list, see help(sc_ptdata.select_data).
            savepath: full path (directories and filename with extension) to save as PNG
            **kwargs: selection to display with keywords, see help(sc_ptdata.select_data).
    '''
    def xticks_minsec(n_frames,fps,interval_sec=20,start_sec=0):
        '''
        Convert xticks expressed in frames to format "minutes:seconds".
        '''
        interval_f = interval_sec*fps
        xticks_loc = list(range(round(start_sec),round(n_frames),round(interval_f)))
        if xticks_loc[-1] > (n_frames-interval_f/2): xticks_loc[-1] = n_frames
        else: xticks_loc.append(n_frames)
        xticks_lbl = []
        for f in xticks_loc: xticks_lbl.append( sc_utils.frames_to_minsec_str(f,fps) )
        return xticks_loc, xticks_lbl

    def xticks_percent(x_percent,length_x):
        '''
        Make xticks as percentage.
        '''
        frac = 100/float(x_percent)
        length_section = round(length_x / frac)
        x_ticks = [t*length_section for t in range(round(frac)+1)]
        x_labels = [ round(t*x_percent) for t in range(round(frac)+1)]
        plt.xticks(x_ticks,x_labels)

    def overlay_vlines(ax, loc, vlattr, numcolour='k', num_hvoffset=None):
        '''
        Args:
            ax: pyplot axis object where to overlay vertical lines.
            loc: list with the location of the lines, in horizontal axis units.
            vlatrr: string with one character for each of these:
                    colour, style, width, f (full vertical) or b (bits at the top and bottom).
                    For example: 'r:2f' means red, dotted, width=2, full vertical line.
            Optional:
                numcolor: Colour for numbers. None for no numbers.
                num_hvoffset: horizontal and vertical offset for the numbers, as percentage of axes lengths.
        Returns:
            none
        '''
        if not vlattr: vlattr='r:2f'
        ylims = ax.get_ylim()
        if vlattr[3] == 'f':
            n_iters = 1
            ymin = [ylims[0]]; ymax = [ylims[1]]
        elif vlattr[3] == 'b':
            n_iters = 2
            the_bit = (abs(ylims[0]) + abs(ylims[1]))*0.1
            ymin = [ylims[0], ylims[1]-the_bit]
            ymax = [ylims[0]+the_bit, ylims[1]]
        else:
            raise Exception(f'Rightmost chracater in "vlattr" should be "f" or "b", but got "{vlattr[3]}".')
        for i in range(n_iters):
            ax.vlines( x = loc, ymin=ymin[i], ymax=ymax[i],
                       colors = vlattr[0],
                       linestyles = vlattr[1],
                       linewidths = int(vlattr[2]) )
        if numcolour:
            hv_offset = [0,0]
            n_sec = len(loc)
            if num_hvoffset:
                xlims = ax.get_xlim()
                h_unit = (xlims[1] - xlims[0])/n_sec
                hv_offset[0] = h_unit * num_hvoffset[0]
                v_unit = ylims[1] - ylims[0]
                hv_offset[1] = v_unit * num_hvoffset[1]
            for i_s in range(n_sec):
                ax.text( loc[i_s] + hv_offset[0],
                         ylims[0] + hv_offset[1],
                         i_s, rotation=0, color=numcolour,
                         horizontalalignment='center')

    kwargs = {**ptdata.vis,**kwargs}
    vistype = kwargs.pop('vistype','line')
    groupby = kwargs.pop('groupby','default')
    y_max = kwargs.pop('y_max',None)
    vscale = kwargs.pop('vscale',1)
    dlattr = kwargs.pop('dlattr',None)
    sections = kwargs.pop('sections',True)
    vlattr = kwargs.pop('vlattr','k:2f')
    snum_hvoff = kwargs.pop('snum_hvoff',[0,1.13])
    y_label = kwargs.pop('y_label','default')
    y_ticks = kwargs.pop('y_ticks',None)
    x_ticklabelling = kwargs.pop('x_ticklabelling','s')
    figtitle = kwargs.pop('figtitle',None)
    axes = kwargs.pop('axes',-1)
    sel_list = kwargs.pop('sel_list',None)
    savepath = kwargs.pop('savepath',None)

    if y_label == 'default': ylabel = ptdata.labels.main
    else: ylabel = y_label
    dlattr_ = [None,None,0.6]
    if dlattr is not None:
        dstr = ''
        for c in dlattr:
            if c.isalpha(): dlattr_[0] = c
            elif c.isdigit() or (c == '.'): dstr += c
            else: dlattr_[1] = c
        if dstr: dlattr_[2] = float(dstr)
    ptdata = select(ptdata,sel_list,**kwargs)
    data_dict = ptdata.data
    data_dict_keys = tuple(data_dict.keys())
    if not isinstance(axes,list): axes = [axes]
    lastaxis_lbl = ptdata.names.dim[axes[-1]]
# TO-DO: case no sections in annotations
    sections_lastaxis_exist = f'trimmed_sections_{lastaxis_lbl}s' in ptdata.topinfo.columns
    if (lastaxis_lbl in kwargs) and sections and sections_lastaxis_exist:
        print(f'Warning: sections are disabled for {lastaxis_lbl} selection.')
        sections = False
# TO-DO:
    # select time maybe by xlims instead of select_data.
    # sections could follow frame selection.
    if figtitle is None: figtitle = ptdata.names.main
    super_title = ''
    if vistype == 'line':
        if (groupby == 'default') and (data_dict[data_dict_keys[0]].ndim > len(axes)):
            groupby = [0]
    elif vistype in ('spectrogram','imshow'):
        if 'spectrogram' in vistype:
            super_title = 'Frequency Spectrum\n'
        ylabel = 'Hz'
    if groupby == 'default':
        if 'imshow' in vistype: groupby = -2
        else: groupby = None
    spt_y = 1
    if sections and sections_lastaxis_exist: spt_y = snum_hvoff[1]*1.1
    data_shape = list(data_dict[data_dict_keys[0]].shape)
    sing_dims = False
    i_1 = []
    if 1 in data_shape: # singleton dimensions
        i_1 = [i for i, x in enumerate(data_shape) if x == 1]
        if not isinstance(groupby,list): groupby = [groupby]
        groupby.extend(i_1)
        sing_dims = True
    if groupby is None: groupby = axes
    if not isinstance(groupby,list): groupby = [groupby]
    groupby = groupby + axes
    for i,v in enumerate(groupby):
        if v<0: groupby[i] = data_dict[data_dict_keys[0]].ndim+v
    groupby = list(dict.fromkeys(groupby)) # remove redundancy
    for i,gb in enumerate(groupby):
        if isinstance(gb,str):
            groupby[i] = ptdata.names.dim.index(gb)
    s = data_shape
    for ii in sorted(groupby, reverse=True):
        if s: del s[ii]
    if len(s) > 1: raise Exception(   'Maximum 2 dimensions allowed for display, ' \
                                    + 'therefore groupby should be for more dimensions.' )
    default_xtick_percentage_str = '25%'
    if x_ticklabelling is None:
        if 'isochronal' in figtitle:
            x_ticklabelling = default_xtick_percentage_str
    else:
        if ('%' not in x_ticklabelling) and ('isochronal' in figtitle):
            x_ticklabelling = default_xtick_percentage_str
    if x_ticklabelling is None:
        xlabel = ''
        x_ticklabelling = 'blank'
    elif x_ticklabelling == 's':
        xlabel = 'time (m:s)'
    elif 'dim' in x_ticklabelling:
        xlabel = ptdata.labels.dim[int(x_ticklabelling.split(' ')[1])]
    elif '%' in x_ticklabelling:
        xlabel = '%'
        xpercent = float(x_ticklabelling.replace('%',''))
    n_sel_top = len(data_dict)
    n_sp = int(np.prod(s)*n_sel_top)
    fig_height = n_sp*2.4*vscale
    i_sp = 1
    fig = plt.figure(figsize=(12,fig_height))
    if y_ticks:
        sp_yticks = []
        sp_axes = []
    for i_top in range(n_sel_top):
        fps = ptdata.topinfo['fps'].iloc[i_top]
        top_arr = data_dict[data_dict_keys[i_top]]
        new_i_top = True
        array_iterator = sc_ndarr.iter( top_arr, lockdim=groupby )
        for vis_arr,i_ch,i_nd in array_iterator:
            if new_i_top: sp_title = '"'+ptdata.topinfo['Name'].iloc[i_top]+'"'
            else: sp_title = ''
            plt.subplot(n_sp,1,i_sp)
            if sing_dims: vis_arr = np.squeeze(vis_arr)
            if vistype in ('line','imshow'):
                len_lastdim = vis_arr.shape[-1]
                if 'line' in vistype:
                    plt.plot( vis_arr.T, color=dlattr_[0], linestyle=dlattr_[1],
                              linewidth=dlattr_[2] )
                elif 'imshow' in vistype:
                    if vis_arr.ndim != 2:
                        check_gb = abs(2-vis_arr.ndim)
                        raise Exception(''.join(['Number of dimensions for imshow is ',\
                                                 f'{vis_arr.ndim} but should be 2. To correct it ',\
                                                 'please check keyword argument groupby has ',
                                                 f'length = {check_gb}']))
                    plt.imshow(vis_arr,aspect='auto')
                    plt.gca().invert_yaxis()
                if x_ticklabelling == 's':
                    plt.xticks( *xticks_minsec(len_lastdim,fps) )
                elif xlabel == '%':
                    xticks_percent(xpercent,vis_arr.shape[axes[-1]])
                plt.xlim((0,len_lastdim))
            elif 'spectrogram' in vistype:
                _,_,t,_ = plt.specgram(vis_arr,Fs=fps,detrend='linear',scale='linear')
                if x_ticklabelling == 's':
                    plt.xticks( *xticks_minsec(t[-1],1,start_sec=t[0]) )
                elif xlabel == '%':
                    xticks_percent(xpercent,len(t))
            if x_ticklabelling == 'blank':
                plt.xticks([],[])
            if y_max: plt.ylim((None,y_max))
            if y_ticks and (vis_arr.ndim == 2):
                if isinstance(y_ticks,list): sp_yticks.append(y_ticks)
                elif isinstance(y_ticks,dict): sp_yticks.append(y_ticks[ data_dict_keys[i_top] ])
                sp_axes.append( plt.gca() )
            plt.ylabel(ylabel)
            if sections and sections_lastaxis_exist:
                overlay_vlines( plt.gca(), ptdata.topinfo['trimmed_sections_frames'].iloc[i_top],
                                vlattr, numcolour=[0.6,0.1,0.2], num_hvoffset=snum_hvoff )
            for i in i_ch:
                if isinstance(ptdata.labels.dimel[i],dict): # dict: different labels for each top array
                    sp_lbl = ptdata.labels.dimel[i][i_top][i_nd[i]]
                elif isinstance(ptdata.labels.dimel[i],list):
                    sp_lbl = ptdata.labels.dimel[i][i_nd[i]] # list: same labels for all top arrays
                else:
                    sp_lbl = ptdata.labels.dimel[i] # (singleton dim.) same labels for all top arrays
                sp_title = ''.join([sp_title,'\n',sp_lbl])
            plt.title(sp_title,y=spt_y)
            if new_i_top: new_i_top = False
            i_sp += 1
    fig.supxlabel(xlabel)
    plt.suptitle( super_title + figtitle , fontsize=16 )
    plt.tight_layout(rect=[0, 0.005, 1, 0.98])
    if y_ticks and (vis_arr.ndim == 2): # TO-DO: this leaves a bit too much space in between ticks
        for i_ax,spax in enumerate(sp_axes):
            yticks_loc = spax.get_yticks()
            if min(yticks_loc) < 0: yticks_loc = np.delete(yticks_loc,0)
            if max(yticks_loc) > (len(sp_yticks[i_ax])-1): yticks_loc = np.delete(yticks_loc,-1)
            yticks_lbl = [ str(round(sp_yticks[i_ax][int(i)],1)).rstrip('0').rstrip('.') \
                           for i in yticks_loc ]
            max_ytick = len(sp_yticks[i_ax])-1
            if max_ytick not in yticks_loc:
                yticks_loc = np.append(yticks_loc,max_ytick)
                yticks_lbl.append(str(round(sp_yticks[i_ax][-1],1)).rstrip('0').rstrip('.'))
                if (yticks_loc[-1] - yticks_loc[-2]) <= 1:
                    yticks_loc = np.delete(yticks_loc,-2)
                    del yticks_lbl[-2]
            sp_axes[i_ax].set_yticks(yticks_loc,labels=yticks_lbl)
    if savepath:
        plt.savefig(savepath + '.png')
