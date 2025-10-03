'''Data class PtData and functions that have ptdata objects as input.'''

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fftfreq
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

from . import ndarr, utils

# .............................................................................
# DATA HANDLING:

class PtData:
    '''
    Data-oriented class.
    Attributes:
        names:
            names.main (str): Main name of the object. Descriptive.
                              It may grow as the names of processes to the data are added.
            names.dim (list[str]): Dimensions of data. Concise, used for selection.
        labels:
            labels.main (str): Main label of the object. Concise or abbreviated.
            labels.dim  (list[str]): Dimensions of data. Concise or abbreviated.
                                     Used for visualisations.
            labels.dimel (list[str]): Elements of dimensions. It can also be a dict of lists with
                                      one item per data array, if the arrays are inhomogeneous.
                                      Used for visualisations.
        data (dict): Each item should be an N-D data array (type numpy.ndarray), and is at the top
                     of the data hierarchy. Its order and index should directly correspond to the
                     index of topinfo.
        topinfo (pandas.DataFrame): Contains information for each entry at the top level.
                                    See documentation for syncoord.utils.load_data
        vis (dict): Arguments for default visualisation parameters. Used when the PtData object is
                    passsed to syncoord.ptdata.visualise
        other (dict): Intialised empty, for any other information.
    Methods:
        print_shape(): Prints shape of data arrays.
        print(): Prints attributes and shape of data arrays.
        checkdim(verbose=True): Checks consistency of dimensions of data arrays (shape),
                                except last dimension.
                                Returns -1 if empty, 0 if inconsistent, 1 if consistent.
        copy(*arg): Returns a deep copy of the object. Optional: 'nodata' to exclude data.
        select(ptdata,*args,**kwargs): Return a selection. See help(syyncord.ptdata.select)
        visualise(**kwargs): Calls syncoord.ptdata.visualise
    Note:
        Use command "vars" to see the content of subfields.
        Examples:
            ptdata = PtData(topinfo)   # initialise data object
            ptdata.names.main = 'Juan' # assign main name
            vars(ptdata.names)
            returns: {'main': 'Juan'}
    Args:
        topinfo (pandas.DataFrame): See documentation for syncoord.utils.load_data
    '''
    def __init__(self,topinfo=None):
        self.names = utils.SubField()
        self.names.main = ''
        self.names.dim = []
        self.labels = utils.SubField()
        self.labels.main = ''
        self.labels.dim = []
        self.labels.dimel = []
        self.data = {}
        self.topinfo = topinfo
        self.vis = {}
        self.other = {}

    def copy(self,*arg):
        _self = deepcopy(self)
        if arg and arg[0] == 'nodata': _self.data = {}
        return _self

    def select(self,*args,**kwargs):
        return select(self,*args,**kwargs)

    def print_shape(self):
        print('data:')
        for k in self.data: print(f'key = {k}, shape = {self.data[k].shape}')
        print()

    def print(self,*arg):
        print(f'names:\n{vars(self.names)}\n')
        print(f'labels:\n{vars(self.labels)}\n')
        if arg and arg[0] == 'nodata': print_data = False
        else: print_data = True
        if print_data and self.data: self.print_shape()
        if self.vis: print(f'vis:\n{self.vis}\n')
        if self.other: print(f'other:\n{self.other}\n')

    def checkdim(self,verbose=True):
        if self.data:
            if len(self.data)==1:
                if verbose:
                    print('Field "data" contains one array.')
                    self.print_shape()
                return 1
            else:
                s_1 = self.data[list(self.data)[0]].shape[:-1]
                for arr in self.data.values():
                    s_2 = arr.shape[:-1]
                    if s_1 != s_1:
                        if verbose:
                            print(''.join(['Inconsistent data dimensions; ',
                                            'only the last may be different.']))
                            self.print_shape()
                        return 0
                return 1
        else:
            if verbose: print('Field "data" is empty.')
            return -1

    def visualise(self,**kwargs):
        retspax = kwargs.get('retspax',False)
        if retspax is True: return visualise(self,**kwargs)
        else: visualise(self,**kwargs)

def load( preproc_data, *props, annot_path=None, max_n_files=None,
              print_info=True, **kwargs ):
    '''
    Load data.
    Args:
        preproc_data (str,dict,numpy.ndarray):
                      If str: Folder with parquet files for preprocesed data
                              (e.g., r"~/preprocessed"),
                              or "make" to produce synthetic data with default values.
                      If dict: As returned by syncoord.utils.testdatavars
                      If numpy.ndarray: As returned by syncoord.utils.init_testdata
        props (str,dict): Path for properties CSV file (e.g., r"~/properties.csv"), or dict
                          with properties. See documentation for syncoord.utils.load_data
                          Optional or ignored if preproc_data = "make"
        Optional:
            annot_path (str): Path and filename for annotations CSV file
                              (e.g., r"~/String_Quartet_annot.csv").
            max_n_files (int): Number of files to load, otherwise loads all files in folder.
            print_info (bool): Print durations of data.
            **kwargs: Passed to syncoord.utils.load_data
                      and syncoord.utils.init_testdatavars if preproc_data = "make"
    Returns:
        PtData object with loaded data.
    '''
    # TO-DO: get main name and label from preprocessed data file
    load_out = utils.load_data( preproc_data, props, annot_path=annot_path,
                                max_n_files=max_n_files, print_info=print_info, **kwargs )

    pos = PtData(load_out[0])
    pos.names.main = ''
    pos.names.dim = load_out[1]
    pos.labels.main = ''
    pos.labels.dim = load_out[2]
    pos.labels.dimel = load_out[3]
    pos.data = load_out[4]
    pos.vis['y_label'] = None
    pos.vis['dlattr'] = '1.2'
    return pos

def select( ptdata,*args,**kwargs ):
    '''
    Select data from a PtData object.
    Args:
        ptdata (PtData): Data object.
        Either of these:
            - List of location for top-level and dimensions, e.g. [top-level, dim 1, dim 2, ...]
            - Arbitrary number of keywords and values, separated or in a dictionary.
                      Keywords are 'top' and dimension names.
            In both cases values are locations or index. They may be 'all', int,
            or slice(start,end), separated or in a nested list with non-continguous values.
    Returns:
        PtData object containing the selected data and topinfo.
    Examples:
        sel = select_data(ptdata, [1,slice(0,180),1,'all'])
        sel = select_data(ptdata, top = [0,1],  point = 1, frame = slice(0,600))
        sel = select_data(ptdata, {'top' : 0, 'point' : 1, 'frame' : slice(0,600)})
        # non-contiguous values for dim 1:
            sel = select_data(ptdata, [1,[slice(0,180),slice(300,600)],1,'all'])
        # if frame is dim -1, keword-value "frame = 'all'" is redundant:
            sel = select_data(ptdata, top = 2, point = 1, frame = 'all')
    Note: top=int for 0-based index; top=[int] or top=[str] to use the key of the data array.
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

    def _checktype(obj):
        allowed_types = ( isinstance(obj,str), isinstance(obj,int),
                          isinstance(obj,slice), isinstance(obj,list) )
        if not any(allowed_types):
            raise Exception( f'Selection type is {type(obj)} but should be either '
                             +'str, int, slice, or list.')

    idx_top_sel = None
    loc_dim = None
    if name_sel:
        top_names = ['top'] + list(ptdata.topinfo.columns)
        valid_names = list(ptdata.names.dim) + top_names
        other_keys = [k for k in name_sel.keys() if k not in valid_names]
        if other_keys: raise Exception(f'Invalid keys: {", ".join(other_keys)}')
        name_sel = {k:v for k,v in name_sel.items() if k in valid_names}
        for k in name_sel: _checktype(name_sel[k])
        if 'top' in name_sel:
            idx_top_sel = name_sel.pop('top')
        loc_dim = [None for _ in range(len(ptdata.names.dim))]
        for i_dim,dim_name in enumerate(ptdata.names.dim):
            if dim_name in name_sel:
                dim_val = name_sel[dim_name]
                if isinstance(dim_val,str):
                    dim_val = ptdata.labels.dimel[i_dim].index(dim_val)
                loc_dim[i_dim] = dim_val
            else: loc_dim[i_dim] = 'all'

    if loc_sel:
        for el in loc_sel: _checktype(el)
        idx_top_sel = loc_sel[0]
        if len(loc_sel) > 1:
            loc_dim = loc_sel[1:]

    if idx_top_sel is not None:
        if idx_top_sel != 'all':
            if isinstance(idx_top_sel,slice):
                sel.topinfo = ptdata.topinfo.iloc[idx_top_sel,:]
            elif isinstance(idx_top_sel,list):
                sel.topinfo = sel.topinfo[sel.topinfo.index.isin(idx_top_sel)]
            else:
                sel.topinfo = ptdata.topinfo.iloc[idx_top_sel:idx_top_sel+1,:]
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
        arr_nd = ptdata.data[i_top]
        if loc_dim and ('all' in loc_dim):
            for i in range_loc_dim:
                if loc_dim[i] == 'all': sel_dim[i] = slice(0,arr_nd.shape[i])
                else: sel_dim[i] = loc_dim[i]
        sel_data_dict[i_top] = arr_nd[tuple(sel_dim)]

    sel.names.main = ptdata.names.main
    sel.labels.main = ptdata.labels.main
    sel.names.dim = []
    sel.labels.dim = []
    sel.labels.dimel = []
    sel.data = sel_data_dict
    sel.vis = deepcopy(ptdata.vis)
    sel.other = deepcopy(ptdata.other)

    # update information:
    for i,dim in enumerate(sel_dim):
        # print('i =',i)
        if isinstance(dim,slice):
            sel.names.dim.append(ptdata.names.dim[i])
            sel.labels.dim.append(ptdata.labels.dim[i])
            if ptdata.labels.dimel[i]:
                if isinstance(ptdata.labels.dimel[i],str):
                    sel.labels.dimel.append(ptdata.labels.dimel[i])
                else: sel.labels.dimel.append(ptdata.labels.dimel[i].copy())
            else: sel.labels.dimel.append(ptdata.labels.dimel[i])

        # update information:
        to_update = [] # data to update
        try: to_update.append( sel.labels.dimel[i] ) # 1 top-level data
        except: pass
        if 'y_ticks' in sel.vis:
            to_update.append( sel.vis['y_ticks'] ) # n top-level data
        if 'freq_bins' in sel.other:
            to_update.append( sel.other['freq_bins'] ) # n top-level data
        for tu in to_update: # this block uses references
        # dimensions:
            if isinstance(loc_dim[i],slice):
                if isinstance(tu,list): # 1 top-level data
                    tu = tu[loc_dim[i]]
                elif isinstance(tu,dict): # n top-level data
                    for k in sel.data.keys():
                        tu[k] = tu[k][loc_dim[i]]
        # tops:
            if (idx_top_sel is not None) and (idx_top_sel != 'all'):
                if isinstance(tu,dict):
                    tu_keys = list(tu.keys())
                    for k in tu_keys:
                        try:
                           if k not in idx_top_sel: del tu[k]
                        except TypeError:
                           if k != idx_top_sel: del tu[k]

    # update title:
    if (arr_nd.ndim > 1) and (arr_nd.shape[-2] > 1) \
    and (isinstance(loc_dim[-2],int)):
        if isinstance(ptdata.labels.dimel[-2],list):
            subtitle = ptdata.labels.dimel[-2][loc_dim[-2]]
        elif isinstance(ptdata.labels.dimel[-2],dict):
            if len(ptdata.labels.dimel[-2]) == 1: # one point
                k = list(ptdata.labels.dimel[-2].keys())
                subtitle = ptdata.labels.dimel[-2][k[0]][loc_dim[-2]]
            else: # more than one point
                new_dimel_labels = [ nl[loc_dim[-2]] for nl in
                                     list(ptdata.labels.dimel[-2].values()) ]
                if len(set(new_dimel_labels)) == 1:
                    subtitle = new_dimel_labels[0]
                else:
                    subtitle = ''
        sel.names.main = sel.names.main + f'\n{subtitle}'
    return sel

def gensec( ptdata, n, print_info=False, get_lengths=False ):
    '''
    Generate equally spaced sections for each data array (along the last dimension).
    The object will be updated in place.
    Args:
        ptdata (PtData): Data object. A prompt will ask to replace the following columns
            of syncoord.ptdata.topinfo if they exist: ptdata.topinfo['Sections'];
            ptdata.topinfo['trimmed_sections_frames']
        n (int): Number of equally spaced sections, conforming to rounding precision.
        Optional:
            print_info (bool): Print sections' length and difference in frames.
            get_lengths (bool): Return array with sections lengths.
    Returns:
        sec_lengths (numpy.ndarray): Sections'lengths in frames, if get_lengths = True.
    '''
    check_1 = 'Sections' in ptdata.topinfo
    check_2 = 'trimmed_sections_frames' in ptdata.topinfo
    if check_1 or check_2:
        ask_again = True
        while ask_again:
            response = input("".join(["Columns 'Section' or 'trimmed_sections_frames' exist in ,",
                                      "ptdata.topinfo\n Do you want to replace them? (y,[n])\n"]))
            if not response: response = 'n'
            if response in ['y','n']: ask_again = False
            else:
                print('Invalid response. Valid responses are: "n" or enter for no; "y" for yes.\n')
        if response in ('n',''):
            print('gensec - warning: function exited without updating ptdata.topinfo')
            return
    all_equal_sex_f = []
    d_keys = list(ptdata.data.keys())
    info_title = True
    if get_lengths: sec_lengths = []
    for i,k in enumerate(ptdata.topinfo.index):
        if k != d_keys[i]:
            raise Exception("".join([f"ptdata.topinfo.index[{k}] doesn't match ",
                                     f"list(ptdata.data.keys())[{i}]"]))
        length_sec = ptdata.data[k].shape[-1]/n
        equal_sex_f = [round(i*length_sec) for i in range(1,n)]
        all_equal_sex_f.append(equal_sex_f)
        fps = ptdata.topinfo.loc[k,'fps']
        if print_info:
            if info_title:
                print("key; sections' length (frames); difference (frames):")
                info_title = False
            ss = [0] + equal_sex_f + [ptdata.data[k].shape[-1]]
            sl = [ ss[i+1]-ss[i] for i in range(len(ss)-1) ]
            d = [ sl[i+1]-sl[i] for i in range(len(sl)-1) ]
            print(f'  {k};  {sl};  {d}')
            if get_lengths: sec_lengths.append(sl)
    ptdata.topinfo['trimmed_sections_frames'] = all_equal_sex_f
    if get_lengths: return np.array(sec_lengths)

# .............................................................................
# ANALYSIS-ORIENTED OPERATIONS:

def norm( ptdata, order, axis=-2 ):
    '''
    Compute norm.
    Wrapper for numpy.linalg.norm
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
        order (int,str): 1 = taxicab, 2 = euclidean. More in documentation for numpy.linalg.norm
        axis (int): Axis (dimension) along wihich to apply vector norms. Default = -2
    Returns:
        New PtData object.
    '''
    norm_ptd = apply( ptdata, np.linalg.norm, ord=order, axis=axis )
    del norm_ptd.names.dim[axis]
    del norm_ptd.labels.dim[axis]
    del norm_ptd.labels.dimel[axis]
    return norm_ptd

def smooth( ptdata,**kwargs ):
    '''
    Apply filter row-wise, to a dimension of N-D arrays (default is last dimension).
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
        Optional kwargs:
            filter_type (str): 'savgol' (default), or 'butter'
            axis (int,str): Default = -1
                Note: Axis is a dimension of the N-D array, specified by index or name.
                      The rightmost axis (-1) is the fastest changing.
            If filter_type = 'savgol':
                window_size (float,list[float]): (seconds)
            If filter_type = 'butter':
                freq_response (str): 'lowpass' (LPF),'highpass' (HPF), or 'bandpass' (BPF).
                cutoff_freq (float,list[float]): cutoff (LPF and HPF) or center frequency (BPF) (Hz).
                bandwidth (float): Only for 'bandpass' (Hz).
            If filter_type = 'savgol' or 'butter':
                order (int): Polynomial order to fit samples (savgol) or half filter order (butter).
                             If filter_type = 'savgol' and order = 1, the result will be the same
                             as a moving average, except that the borders will be interpolated.
    Returns:
        New PtData object.
    '''
    filter_type = kwargs.get('filter_type','savgol')
    if filter_type is None: return ptdata
    freq_response = kwargs.get('freq_response','lowpass')
    cutoff_freq = kwargs.get('cutoff_freq',2)
    order = kwargs.get('order',3)
    window_size = kwargs.get('window_size',3)
    bandwidth = kwargs.get('bandwidth',None)
    axis = kwargs.get('axis',-1)

    if (filter_type=='butter') and (freq_response=='bandpass') and (bandwidth is None):
        raise Exception('bandwidth is missing')
    if isinstance(axis,str):
        axis_lbl = axis
        axis = ptdata.names.dim.index(axis)
    elif isinstance(axis,int): axis_lbl = ptdata.names.dim[axis]
    else: raise Exception('axis should be either str or int')

    if filter_type == 'butter':
        if 'band' in freq_response:
            if bandwidth:
                bw_half = bandwidth/2
                bp_freq = []
                for f in cutoff_freq:
                    if isinstance(f,list):
                        raise Exception( 'When bandwidth is specified, the elements of \
                                          cutoff_freq should be scalars, not nested lists.' )
                    else:
                        bp_freq.append( [v if (v>0) else 0.001 for v in [f-bw_half,f+bw_half]] )
                cutoff_freq = bp_freq
            else:
                if (len(cutoff_freq) == 2) \
                and (not isinstance(cutoff_freq[0],list)) \
                and (not isinstance(cutoff_freq[1],list)): cutoff_freq = [cutoff_freq]
        multiband_param = cutoff_freq
        main_name =  f'Filtered with Butterworth {freq_response}'
        if ptdata.names.main: main_name = f'{main_name}\n{ptdata.names.main}'
        other = dict(list(kwargs.items())[:4]+list(kwargs.items())[5:])
        def _sosfiltfilt(arr,sos):
            return signal.sosfiltfilt(sos, arr)
    elif filter_type in ['savgol','mean']:
        multiband_param = window_size
        if filter_type == 'savgol':
            main_name =  f'Filtered with Savitzky-Golay\n{ptdata.names.main}'
            def _savgol(arr,ws):
                return signal._savgolfilter( arr, ws, order)
        elif filter_type == 'mean':
            # mean (moving average) disabled until topinfo start and sections are offset when
            # mode = 'valid'
            raise Exception("filter_type = 'mean' option not available")
            mode = kwargs.get('mode','same')
            main_name =  f'Filtered with moving mean)\n{ptdata.names.main}'
            def _mean(arr,ws,mode):
                wsr = round(ws)
                return np.convolve( arr, np.ones(wsr)/wsr, mode=mode)
        other = dict(list(kwargs.items())[4:])
    if isinstance(multiband_param,np.ndarray): pass
    elif isinstance(multiband_param,tuple): multiband_param = list(multiband_param)
    elif not isinstance(multiband_param,list): multiband_param = [multiband_param]
    n_mbp = len(multiband_param)

    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    vis_opt = {**ptdata.vis,'dlattr':'1.2'}
    if n_mbp > 1:
        means = [ round(np.mean(f).item(),3) for f in multiband_param ]
        vis_opt['y_ticks'] = means
        if filter_type == 'butter':
            dim_names.insert(axis,'frequency')
            dim_labels.insert(axis,'freq.')
            dimel_labels.insert(axis, [f'{m}Hz' for m in means] )
        elif filter_type in ['savgol','mean']:
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
                sos = signal.butter(order, mbp, freq_response, fs=fps, output='sos')
                arr_out = np.apply_along_axis(_sosfiltfilt, axis,  dd_in[i_top], sos)
            else:
                if filter_type == 'savgol':
                    arr_out = np.apply_along_axis(_savgol, axis, dd_in[i_top], int(mbp*fps))
                if filter_type == 'mean':
                    arr_out = np.apply_along_axis(_mean, axis, dd_in[i_top], int(mbp*fps), mode)
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

def tder( ptdata, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.tder
    '''
    return apply( ptdata, ndarr.tder, **kwargs )

def speed( ptdata, **kwargs ):
    '''
    Another wrapper for syncoord.ndarr.tder but order is always 1
    '''
    return apply( ptdata, ndarr.tder, order=1, **kwargs )

def peaks_to_phase( ptdata, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.peaks_to_phase
    Argument "min_dist" (seconds) along with the corresponding fps replace agrument
    "distance" (frames) of syncoord.ndarr.peaks_to_phase
    Args:
        min_dist (float): Minimum distance in seconds.
        **kwargs passed to syncoord.ndarr.peaks_to_phase
    '''
    min_dist = kwargs.pop('min_dist')
    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in:
        kwargs['distance'] = min_dist * ptdata.topinfo.loc[k,'fps']
        dd_out[k] = ndarr.peaks_to_phase( dd_in[k], **kwargs )

    pkphi = PtData(ptdata.topinfo)
    pkphi.names.main = 'Peaks Phase'
    pkphi.names.dim = deepcopy(ptdata.names.dim)
    pkphi.labels.main = r'$\phi$'
    pkphi.labels.dim = deepcopy(ptdata.labels.dim)
    pkphi.labels.dimel = deepcopy(ptdata.labels.dimel)
    pkphi.data = dd_out
    pkphi.vis = {**ptdata.vis,'dlattr':'k0.8', 'vlattr':'r:3f','vistype':'line'}
    pkphi.other = deepcopy(ptdata.other)
    return pkphi

def kuramoto_r( ptdata ):
    '''
    Wrapper for syncoord.ndarr.kuramoto_r
    '''
    return apply( ptdata, ndarr.kuramoto_r )

def fourier( ptdata, window_duration, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.fourier_transform
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
        window_duration (float): Length of the FFT window (seconds) unless fps = None.
        Optional:
            **kwargs: Input parameters. See documentation for syncoord.ndarr.fourier_transform
            Note: If mode='valid', the sections (ptdata.topinfo['Sections'] and
                  ptdata.topinfo['trimmed_sections_frames']) will be shifted accordingly.
    Returns:
        New PtData object.
    '''
    mode = kwargs.get('mode')

    if mode and (mode=='valid'):
        topinfo = utils.trim_topinfo_start(ptdata,window_duration/2)
    else:
        topinfo = ptdata.topinfo

    first_fbin = kwargs.get('first_fbin',1)
    freq_bins = {}
    freq_bins_labels = {}
    wl = window_duration

    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in:
        fps = ptdata.topinfo.loc[k,'fps']
        if 'fps' not in kwargs: wl = round(window_duration * fps)
        dd_out[k] = ndarr.fourier_transform(dd_in[k], wl,**kwargs)
        freq_bins[k] = np.abs(( fftfreq(wl)*fps )[first_fbin:np.floor(wl/2 + 1).astype(int)])
        freq_bins_rounded = np.round(freq_bins[k],2)
        rdif = abs(np.mean(freq_bins_rounded-np.round(freq_bins_rounded)))
        if rdif < 0.001: freq_bins_rounded = np.round(freq_bins_rounded,0).astype(int)
        freq_bins_labels[k] = [f'bin {i}: {f} Hz' for i,f in enumerate(freq_bins_rounded)]

    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    dimel_labels.insert(-1,freq_bins_labels)
    vis = {'dlattr':'k0.8','vlattr':'r:3f'}
    if ('output' in kwargs) and (kwargs['output'] == 'phase'):
        main_name = 'Phase'
        main_label = r'$\phi$'
        dim_labels.insert(-1,'freq.')
        vis['vistype'] = 'line'
    else:
        main_name = 'Frequency Spectrum'
        if kwargs['output'] == 'amplitude':
            main_name = f'{main_name} (amplitude)'
        main_label = 'Spectrum'
        dim_labels.insert(-1,'freq.')
        vis['vistype'] = 'imshow'
    dim_names.insert(-1,'frequency')
    vis['y_ticks'] = freq_bins
    other = {'freq_bins':freq_bins}

    fft_result = PtData(topinfo)
    fft_result.names.main = main_name
    fft_result.names.dim = dim_names
    fft_result.labels.main = main_label
    fft_result.labels.dim = dim_labels
    fft_result.labels.dimel = dimel_labels
    fft_result.data = dd_out
    fft_result.vis = vis
    fft_result.other = other
    return fft_result

def plv( ptdata, windows, window_hop=None, pairs_axis=0,
            fixed_axes=None, plv_axis=-1, mode='same', verbose=False ):
    '''
    Pairwise Phase-Locking Values upon moving window or sections.
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
                N-D arrays should have at least 2 dimensions.
        windows (float,str): Window length in seconds for sliding window or 'sections'.
        Optional:
            window_hop (float): Sliding window's step in seconds. None for a step of 1 frame.
            pairs_axis (int): Dimension to form the pairs.
            fixed_axes (int,list[int]): Dimension or dimensions passed to the windowed PLV function.
                       Default is [-2,-1] if N-D array dimensions are 3 or more; -1 if 2 dimensions.
            plv_axis (int): Dimension to perform the PLV function.
                      For example:
                          data.shape = (4,1,15,9000)
                          pairs_axis = 0:
                              6 pairs to be formed: ([0,1],[0,2],[0,3],[1,2],[1,3],[2,3])
                          fixed_axes = [-2,-1]:
                              Passed to the PLV function: data.shape = (15,9000)
                          plv_axis = -1:
                              The PLV function will be applied across the 9000 points of
                              each of the 15 vectors.
                Note: axis is a dimension of the N-D array.
                      The rightmost axis (-1) is the fastest changing.
            mode (str): 'same' (post-process zero-padded, same size as input) or 'valid'.
                        Note: If mode='valid', the sections (ptdata.topinfo['Sections'] and
                        ptdata.topinfo['trimmed_sections_frames']) will be shifted accordingly.
            verbose (bool): Display progress.
    Returns:
        New PtData object.
    '''
    assert isinstance(windows,(float,int)) or (isinstance(windows,str) and windows=='sections'),\
    'Wrong value for argument "windows".'
    if isinstance(windows,(float,int)):
        slidingw = True
        plv_lbl = ' (sliding window)'
    else:
        slidingw = False
        plv_lbl = ' (sections)'
    if not window_hop: window_step = 1
    else: window_step = None
    new_fps = []
    dd_in = ptdata.data
    dd_out = {}
    c = 1
    for k in dd_in:
        if verbose:
            print(f'processing array {k} ({c} of {len(ptdata.data.keys())})')
            c+=1
        if fixed_axes is None:
            if dd_in[k].ndim > 2:
                fixed_axes = [-2,-1]
            elif dd_in[k].ndim == 2:
                fixed_axes = -1
            if dd_in[k].ndim < 2:
                raise Exception('number of dimensions in data arrays should be at least 2')
        plvkwargs = {}
        fps = ptdata.topinfo.loc[k,'fps']
        if slidingw:
            plvkwargs['window_length'] = round(windows * fps)
            if window_hop:
                plvkwargs['window_step'] = round(window_hop * fps)
                new_fps.append(fps/plvkwargs['window_step'])
            plvkwargs['mode'] = mode
        else:
            sections = ptdata.topinfo.trimmed_sections_frames[k]
            n_frames = dd_in[k].shape[plv_axis]
            new_fps.append( fps * (len(sections)+1) / n_frames )
            plvkwargs['sections'] = sections
        plvkwargs['axis'] = plv_axis
        dd_out[k], pairs_idx, _ = ndarr.apply_to_pairs( dd_in[k], ndarr.plv,
                                                        pairs_axis, fixed_axes=fixed_axes,
                                                        verbose=verbose, **plvkwargs )
    topinfo = ptdata.topinfo
    if slidingw and (mode == 'valid'): topinfo = utils.trim_topinfo_start(ptdata,windows/2)
    if (not slidingw) or (slidingw and window_hop):
        topinfo = deepcopy(topinfo)
        if 'trimmed_sections_frames' in topinfo:
            new_sec = []
            for o,n,s in zip(topinfo['fps'],new_fps,topinfo['trimmed_sections_frames']):
                new_sec.append( [ round(v*n/o) for v in s ] )
        topinfo['trimmed_sections_frames'] = new_sec
        topinfo['fps'] = new_fps

    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    if isinstance(fixed_axes,list): groupby = fixed_axes[0]
    else: groupby = fixed_axes
    i_nlbl = groupby
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

    plvdata = PtData(topinfo)
    plvdata.names.main = 'Pairwise Phase-Locking Value' + plv_lbl
    plvdata.names.dim = dim_names
    plvdata.labels.main = 'PLV'
    plvdata.labels.dim = dim_labels
    plvdata.labels.dimel = dimel_labels
    plvdata.data = dd_out
    plvdata.vis = {'dlattr':'1.2','groupby':groupby, 'vistype':vistype, 'vlattr':'r:3f'}
    if 'freq_bins' in ptdata.other:
        plvdata.vis['y_ticks'] = ptdata.other['freq_bins'].copy()
    if (not slidingw) and (vistype == 'line'):
        plvdata.vis = {**plvdata.vis, 'vistype':'cline', 'sections':False, 'x_ticklabelling':'index'}
    plvdata.other = ptdata.other.copy()
    return plvdata

def wct( ptdata, minmaxf, pairs_axis, fixed_axes, **kwargs ):
    '''
    Pairwise Wavelet Coherence Transform with Morlet wavelet.
    Wrapper for pycwt.wct
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
                         N-D arrays should have at least 2 dimensions.
        minmaxf (list[float]): Minimum and maximum frequency (Hz). Can be the same value.
                               The minimum frequency mimics that of Matlab's function "wcoherence".
        pairs_axis (list): Dimensions to form the pairs.
        fixed_axes (int,list[int]): Dimension(s) passed to the wct function.
        Optional kwargs:
            nspo (float): Number of scales per octave. Default = 12.
            postprocess (str): None = raw WCT (default)
                               'coinan' = the cone of influence (COI) is filled with NaN
            verbose (bool): It will apply to syncoord.ndarr.apply_to_pairs
    Returns:
        New PtData object.
    References:
        https://github.com/regeirk/pycwt
        https://pycwt.readthedocs.io
    '''
    import pycwt

    wct_pairs_kwargs = {}
    wct_pairs_kwargs['fixed_axes'] = fixed_axes
    wct_pairs_kwargs['dj'] = kwargs.get('dj',1/12)
    wct_pairs_kwargs['flambda'] = pycwt.Morlet().flambda()
    wct_pairs_kwargs['postprocess'] = kwargs.get('postprocess',None)
    wct_pairs_kwargs['verbose'] = kwargs.get('verbose',False)

    dd_in = ptdata.data
    dd_out = {}
    c = 1
    for k in dd_in:
        arr_nd = dd_in[k]
        if arr_nd.ndim < 2: raise Exception(f'Data dimensions should be at least 2,\
                                              but currently are {arr_nd.ndim}')
        fps = ptdata.topinfo.loc[k,'fps']
        pairs_results = ndarr.apply_to_pairs( arr_nd, ndarr.wct, pairs_axis, minmaxf=minmaxf,
                                              fps=fps, **wct_pairs_kwargs )

        dd_out[k] = pairs_results[0]

    pairs_idx = pairs_results[1]
    new_fixed_axes = pairs_results[2]
    freq_bins = pairs_results[3][0][0].tolist()
    if len(freq_bins) == 1: one_freq = True
    else: one_freq = False
    freq_bins_round = np.round(freq_bins,2).tolist()

    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()

    if not isinstance(fixed_axes,list): fixed_axes = [fixed_axes]
    if isinstance(new_fixed_axes,list): groupby = new_fixed_axes[0]
    else: groupby = new_fixed_axes

    if one_freq:
        main_name = f'Wavelet Coherence Spectrum at {round(freq_bins[0],2)} Hz'
        y_ticks = None
    else:
        i_freq_lbl = groupby
        if len(fixed_axes) < len(new_fixed_axes):
            dim_names.insert(i_freq_lbl,'frequency')
            dim_labels.insert(i_freq_lbl,'freq')
            dimel_labels.insert(i_freq_lbl,freq_bins_round)
        else:
            if i_freq_lbl < 0: i_freq_lbl = len(dim_names) + i_freq_lbl
            if (i_freq_lbl != pairs_axis) and (i_freq_lbl >= 0):
                dim_names[i_freq_lbl] = 'frequency'
                dim_labels[i_freq_lbl] = 'freq.'
                dimel_labels[i_freq_lbl] = freq_bins_round
        main_name = 'Wavelet Coherence Spectrum'
        y_ticks = freq_bins_round

    oldnew_faxes_pos = [[],[]]
    for i_fal,fa in enumerate([fixed_axes,new_fixed_axes]):
        for i_fap,a in enumerate(fa):
            if a < 0: oldnew_faxes_pos[i_fal].append( dd_out[0].ndim + a)
            else: oldnew_faxes_pos[i_fal].append( a )
    if not (len(fixed_axes) < len(new_fixed_axes)) \
       and (oldnew_faxes_pos[0] != oldnew_faxes_pos[1]):
        idx_remdim = [v for v in oldnew_faxes_pos[0] if v not in oldnew_faxes_pos[1]]
        for i_rem in idx_remdim:
            del dim_names[i_rem]
            del dim_labels[i_rem]
            del dimel_labels[i_rem]

    dim_names[pairs_axis] = 'pair'
    dim_labels[pairs_axis] = 'pairs'
    k = list(dd_in.keys())[0]
    n_pair_el = dd_in[k].shape[pairs_axis]
    n_pairs = (n_pair_el**2 - n_pair_el)//2
    dimel_labels[pairs_axis] = ['pair '+str(p) for p in pairs_idx]
    if groupby == -1: vistype = 'line'
    else: vistype = 'imshow'

    wctdata = PtData( deepcopy(ptdata.topinfo) )
    wctdata.names.main = main_name
    wctdata.names.dim = dim_names
    wctdata.labels.main = 'WCT'
    wctdata.labels.dim = dim_labels
    wctdata.labels.dimel = dimel_labels
    wctdata.data = dd_out
    wctdata.vis = { 'groupby':groupby, 'vistype':vistype, 'rescale':False,
                    'dlattr':'1.2', 'vlattr':'r:3f' }
    if one_freq: wctdata.vis['vistype'] = 'line'
    if y_ticks: wctdata.vis['y_ticks'] = y_ticks
    wctdata.other['freq_bins'] = freq_bins
    return wctdata

def gxwt( ptdata, minmaxf, pairs_axis, fixed_axes, **kwargs ):
    '''
    Pairwise multi-dimensional cross-wavelet spectrum.
    Wrapper for syncoord.ndarr.gxwt
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
                N-D arrays should have at least 2 dimensions.
        minmaxf (list[float]): Minimum and maximum frequency (Hz).
        pairs_axis (list): Dimensions to form the pairs.
        fixed_axes (int,list[int]): Dimension(s) passed to the gxwt function.
        Optional kwargs:
            Keyword arguments to syncoord.ndarr.gxwt
            verbose (bool): It will apply to syncoord.ndarr.apply_to_pairs and syncoord.ndarr.gxwt
    Returns:
        New PtData object.
    '''
    verbose = kwargs.get('verbose',True)
    if 'matlabeng' in kwargs: neweng = False
    else:
        neweng = True
        extfunc_path = kwargs.pop('extfunc_path',None)
        gxwt_path = kwargs.pop('gxwt_path',None)
        addpaths = [extfunc_path,gxwt_path]
        kwargs['matlabeng'] = utils.matlab_eng(addpaths,verbose)

    dd_in = ptdata.data
    dd_out = {}
    c = 1
    for k in dd_in:
        arr_nd = dd_in[k]
        if arr_nd.ndim < 2: raise Exception(f'Data dimensions should be at least 2,\
                                              but currently are {arr_nd.ndim}')
        fps = ptdata.topinfo.loc[k,'fps']
        pairs_results = ndarr.apply_to_pairs( arr_nd, ndarr.gxwt, pairs_axis,
                                              fixed_axes=fixed_axes,
                                              minmaxf=minmaxf, fps=fps, **kwargs )
        dd_out[k] = pairs_results[0]

    if neweng:
        kwargs['matlabeng'].quit()
        if verbose: print('Disconnected from Matlab.')

    pairs_idx = pairs_results[1]
    new_fixed_axes = pairs_results[2]
    freq_bins = pairs_results[3][0][0].tolist()
    if not isinstance(freq_bins,list): one_freq = True
    else: one_freq = False
    freq_bins_round = np.round(freq_bins,2).tolist()

    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()

    if not isinstance(fixed_axes,list): fixed_axes = [fixed_axes]
    if isinstance(new_fixed_axes,list): groupby = new_fixed_axes[0]
    else: groupby = new_fixed_axes

    if one_freq:
        main_name = f'Generalised Cross-Wavelet Transform at {round(freq_bins,2)} Hz'
        y_ticks = None
    else:
        i_freq_lbl = groupby
        if len(fixed_axes) < len(new_fixed_axes):
            dim_names.insert(i_freq_lbl,'frequency')
            dim_labels.insert(i_freq_lbl,'freq')
            dimel_labels.insert(i_freq_lbl,freq_bins_round)
        else:
            if i_freq_lbl < 0: i_freq_lbl = len(dim_names) + i_freq_lbl
            if (i_freq_lbl != pairs_axis) and (i_freq_lbl >= 0):
                dim_names[i_freq_lbl] = 'frequency'
                dim_labels[i_freq_lbl] = 'freq.'
                dimel_labels[i_freq_lbl] = freq_bins_round
        main_name = 'Generalised Cross-Wavelet Spectrum'
        y_ticks = freq_bins_round

    oldnew_faxes_pos = [[],[]]
    for i_fal,fa in enumerate([fixed_axes,new_fixed_axes]):
        for i_fap,a in enumerate(fa):
            if a < 0: oldnew_faxes_pos[i_fal].append( dd_out[0].ndim + a)
            else: oldnew_faxes_pos[i_fal].append( a )
    if not (len(fixed_axes) < len(new_fixed_axes)) \
       and (oldnew_faxes_pos[0] != oldnew_faxes_pos[1]):
        idx_remdim = [v for v in oldnew_faxes_pos[0] if v not in oldnew_faxes_pos[1]]
        for i_rem in idx_remdim:
            del dim_names[i_rem]
            del dim_labels[i_rem]
            del dimel_labels[i_rem]

    dim_names[pairs_axis] = 'pair'
    dim_labels[pairs_axis] = 'pairs'
    k = list(dd_in.keys())[0]
    n_pair_el = dd_in[k].shape[pairs_axis]
    n_pairs = (n_pair_el**2 - n_pair_el)//2
    dimel_labels[pairs_axis] = ['pair '+str(p) for p in pairs_idx]
    if groupby == -1: vistype = 'line'
    else: vistype = 'imshow'

    xwtdata = PtData( deepcopy(ptdata.topinfo) )
    xwtdata.names.main = main_name
    xwtdata.names.dim = dim_names
    xwtdata.labels.main = 'GXWT'
    xwtdata.labels.dim = dim_labels
    xwtdata.labels.dimel = dimel_labels
    xwtdata.data = dd_out
    xwtdata.vis = { 'groupby':groupby, 'vistype':vistype, 'rescale':True,
                    'dlattr':'1.2', 'vlattr':'r:3f' }
    if y_ticks: xwtdata.vis['y_ticks'] = y_ticks
    xwtdata.other['freq_bins'] = freq_bins
    return xwtdata

def rho( ptdata, exaxes=None, mode='all', method='SynCoord' ):
    '''
    Cluster Phase.
    Wrapper for ndarr.cluster_phase_rho or multiSyncPy.synchrony_metrics.rho
    Args:
        ptdata (PtData): Data object with phase angles.
                See documentation for syncoord.ptdata.PtData
        exaxes (int,None):
                If int: Dimension(s) to exclude from grouping, except last dimension.
                If None: All dimensions except last will be grouped.
                Set to -2 if that dimension's name is 'frequency'.
        Optional:
            mode (str): 'all' or 'mean'
            method (str): 'SynCoord' (default), 'multiSyncPy'
    Returns:
        New PtData object.
    References:
        https://doi.org/10.3389/fphys.2012.00405
        https://github.com/cslab-hub/multiSyncPy
    '''
    if len(ptdata.data)==1: cdv = False
    else: cdv = True
    assert ptdata.checkdim(verbose=cdv)==1

    if mode == 'all':
        i_out = 0
        n_out = mode
    elif mode == 'mean': i_out = n_out = 1
    else: raise Exception('mode can only be "all" or "mean"')

    if method == 'SynCoord': sm_rho = ndarr.cluster_phase_rho
    elif method == 'multiSyncPy': from multiSyncPy.synchrony_metrics import rho as sm_rho
    else: raise Exception('Invalid value for "method".')

    if ptdata.names.dim[-2] == 'frequency': exaxes = -2

    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in:
        dd_out[k] = ndarr.apply_dimgroup( dd_in[k], sm_rho, exaxes=exaxes,
                                          i_out=i_out, n_out=n_out )

    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    idx_grpax, _ = utils.invexaxes(exaxes, dd_in[list(dd_in)[0]].shape)
    del idx_grpax[-1]
    idx_grpax.sort(reverse=True)
    for i in idx_grpax:
        del dim_names[i]
        del dim_labels[i]
        del dimel_labels[i]
    vis = {**ptdata.vis}
    vis['dlattr'] = '1.2'
    if dd_out[list(dd_out)[0]].ndim >= 2: vis = {**vis,'vistype':'imshow'}
    if 'freq_bins' in ptdata.other:
        vis['y_ticks'] = ptdata.other['freq_bins'].copy()

    ptd_rho = PtData(ptdata.topinfo)
    ptd_rho.names.main = r'Cluster Phase $\rho$'
    ptd_rho.names.dim = dim_names
    ptd_rho.labels.main = r"$\rho$"
    ptd_rho.labels.dim = dim_labels
    ptd_rho.labels.dimel = dimel_labels
    ptd_rho.data = dd_out
    ptd_rho.vis = vis
    ptd_rho.other = deepcopy(ptdata.other)
    return ptd_rho

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

def isochrsec( ptdata, last=False, axis=-1 ):
    '''
    Wrapper for ndarr.isochronal_sections.
    Time-rescale data so that it fits into sections of the same length.
    The length of the resulting sections will be the length of the largest input index of sections.
    Args:
        ptdata (PtData): Data object with topinfo containing column 'trimmed_sections_frames'.
                See documentation for syncoord.utils.load_data
        Optional:
            last (bool): If True, the last section is from the last index of sections to the end.
            axis (int): Dimension to apply the process.
                Note: Axis is a dimension of the N-D array.
                      The rightmost axis (-1) is the one that changes most frequently.
    Returns:
        New PtData object, with 'trimmed_sections_frames' modified.
    '''
    data_list = []
    for arr_nd in ptdata.data.values():
        data_list.append(arr_nd)
    idx_sections = ptdata.topinfo['trimmed_sections_frames'].values.tolist()
    isochr_data, idx_isochr_sections = ndarr.isochronal_sections(data_list,idx_sections,last,axis)
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
    isec.vis = {**ptdata.vis,'x_ticklabelling':'%'}
    isec.other = ptdata.other.copy()
    return isec

def aggrsec( ptdata, aggregate_axes=[-2,-1], sections_axis=1,
                    omit=None, function='mean' ):
    '''
    Aggregate sections.
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
                The field topinfo should have columns 'trimmed_sections_frames',
                with lists having equally spaced indices.
                See documentation for syncoord.ptdata.isochrsec
        Optional:
            aggregate_axes (int,list[int]): Dimension(s) to aggregate.
            sections_axis: Index of aggregated dimensions, where the sections are taken from.
                  For example:
                      data.shape = (6,1,15,9000)
                      aggregate_axes = [-2,-1]:
                          shape of aggregated dimensions: (15,9000)
                      sections_axis = 1:
                          The sections are taken from the dimension with length 9000 points.
                Note: Axis is a dimension of the N-D array.
                      The rightmost axis (-1) is the fastest changing.
            omit (int,list[int]): sections to omit
            function (str): 'mean' or 'sum'
    Returns:
        New PtData object.
    '''
    i_sec_ax = aggregate_axes[sections_axis]
    if not isinstance(omit,list): omit = [omit]
    if not isinstance(aggregate_axes,list): aggregate_axes = [aggregate_axes]
    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in:
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

def aggrax( ptdata, axis=0, function='mean' ):
    '''
    Wrapper for numpy.sum or numpy.mean
    Aggregate axes of N-D data arrays.
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
        Optional:
            function (str): 'sum' or 'mean'
            axis (int): Dimension to which the operation will be applied.
                Note: Axis is a dimension of the N-D array.
                      The rightmost axis (-1) is the fastest changing.
    Returns:
        New PtData object.
    '''
    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in:
        if function == 'sum': dd_out[k] = np.nansum(dd_in[k],axis=axis)
        elif function == 'mean':
            dd_out[k] = np.nanmean(dd_in[k],axis=axis)
    main_name = ptdata.names.main
    if function == 'sum':
        function_lbl = 'added'
    elif function == 'mean':
        function_lbl = 'mean'
    aggr_lbl = f'{function_lbl} dim. "{ptdata.names.dim[axis]}"'
    if main_name[-1] == ')': main_name = ''.join([main_name[:-1],', ',aggr_lbl,')'])
    else: main_name = ''.join([main_name,'\n(',aggr_lbl,')'])
    dim_names = deepcopy(ptdata.names.dim)
    del dim_names[axis]
    dim_labels = deepcopy(ptdata.labels.dim)
    del dim_labels[axis]
    dimel_labels = deepcopy(ptdata.labels.dimel)
    del dimel_labels[axis]
    vis = {**ptdata.vis}
    ndim_in = dd_in[next(iter(dd_in))].ndim
    if (axis==-2) or ((ndim_in - axis)==2):
        vis['groupby'] = None
        if vis['vistype'] not in ['line','cline']: vis['vistype'] = 'line'
        if 'sections' in main_name:
            vis = {**vis, 'vistype':'cline','x_ticklabelling':'index','sections':False}
    else: vis['groupby'] = axis
    if len(dim_names) == 1:
        if vis['vistype'] not in ['line','cline']: vis['vistype'] = 'line'
        vis['dlattr'] = '-1'
        vis['groupby'] = 0
    other =  deepcopy(ptdata.other)
    if 'frequency' not in dim_names:
        if 'y_ticks' in vis: del vis['y_ticks']
        if 'freq_bins' in other: del other['freq_bins']

    agg = PtData(ptdata.topinfo)
    agg.names.main = main_name
    agg.names.dim = dim_names
    agg.labels.main = ptdata.labels.main
    agg.labels.dim = dim_labels
    agg.labels.dimel = dimel_labels
    agg.data = dd_out
    agg.vis = vis
    agg.other = other
    return agg

def aggrtop( ptdata, function='mean', axis=0):
    '''
    Aggregate top-level homogeneous N-D data arrays.
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
        Optional:
            function (str): 'sum', 'mean', 'concat', or 'vstack'
            axis (int): Dimension along which the arrays will be joined. Only for 'concat'.
    Returns:
        New PtData object. Property 'topinfo' will retain only columns whose rows are identical,
        collapsed into a single row.
    '''
    dd_in = ptdata.data
    isarray = dd_in[next(iter(dd_in))].ndim
    if function in ['concat','vstack']:
        arr_nd_out = []
        first = True
        for k in dd_in:
            if function == 'vstack':
                if first:
                    arr_nd_out = dd_in[k]
                    first = False
                else:
                    arr_nd_out = np.vstack((arr_nd_out, dd_in[k]))
            else:
                if isarray: arr_nd_out = np.concatenate((arr_nd_out, dd_in[k]), axis)
                else: arr_nd_out.append(dd_in[k])
        arr_nd_out = np.array(arr_nd_out)
        function_lbl = 'Concatenated'
    else:
        arr_nd_out = np.zeros(dd_in[next(iter(dd_in))].shape)
        for k in dd_in: arr_nd_out += dd_in[k]
        if function == 'sum': function_lbl = 'Added'
        elif function == 'mean':
            arr_nd_out = arr_nd_out/len(dd_in)
            function_lbl = 'Mean'
    dd_out = {0:arr_nd_out}

    main_name = ptdata.names.main
    dim_names = deepcopy(ptdata.names.dim)
    dim_labels = deepcopy(ptdata.labels.dim)
    dimel_labels = deepcopy(ptdata.labels.dimel)

    if function == 'vstack':
        dim_names = ['topstack']+dim_names
        dim_labels = ['tops.']+dim_labels
        dimel_labels = ['tops.']+dimel_labels

    aggr_lbl = f'\n({function_lbl} top-level data)'
    colnames_identical = []
    for colname in ptdata.topinfo:
        if len(ptdata.topinfo[colname].drop_duplicates()) == 1:
            colnames_identical.append(colname)
    topinfo = ptdata.topinfo[colnames_identical].iloc[0].copy().to_frame().T.reset_index(drop=True)

    agg = PtData(topinfo)
    agg.names.main = ptdata.names.main + aggr_lbl
    agg.names.dim = dim_names
    agg.labels.main = ptdata.labels.main
    agg.labels.dim = dim_labels
    agg.labels.dimel = dimel_labels
    agg.data = dd_out
    agg.vis = {**ptdata.vis, 'groupby':0, 'sections':True}
    if not isarray: agg.vis['x_ticklabelling'] = 'default'
    agg.other = deepcopy(ptdata.other)
    return agg

def secstats( ptdata, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.section_stats
    Descriptive statistics for sections of N-D data arrays.
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
        Optional:
            last (bool): If True (default), last section is from last sections' index to end.
            margins (int,list[int],dict). Trim at the beginning and ending, in seconds.
                     If scalar: Same trim bor beginning and ending.
                     If list: Trims for beginning and ending. Nested lists for sections.
                     If dict: Items correspond to N-D data arrays, keys are same as ptdata.data
                              (and as in ptdata.topinfo), and values are lists.
            axis (int): Dimension upon which to run the process.
            statnames (str,list[str]): Statistics to compute. Default is all.
                     Available options: 'mean','median','min','max','std'.
            **kwargs: see documentation for syncoord.ndarr.section_stats
    Returns:
        New PtData object.
    '''
    cont = kwargs.get('cont',False)
    if not 'statnames' in kwargs:
        kwargs['statnames'] = [ 'mean','median','min','max','std' ]
    axis = kwargs.get('axis',-1)
    appaxis_lbl = ptdata.names.dim[axis]
    sections_appaxis_exist = f'trimmed_sections_{appaxis_lbl}s' in ptdata.topinfo.columns
    lbl_topinfo_sec = f'trimmed_sections_{appaxis_lbl}s'
    if not sections_appaxis_exist:
        raise Exception(  f"Colummn ''{lbl_topinfo_sec}' for axis {axis} \
                           not found in ptdata.topinfo" )
    margins_dict = ('margins' in kwargs) and isinstance(kwargs['margins'],dict)
    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in:
        idx_sections = ptdata.topinfo[lbl_topinfo_sec].loc[k]
        fps = ptdata.topinfo['fps'].loc[k]
        if margins_dict: kwargs['margins'] = kwargs['margins'][k]
        dd_out[k] = ndarr.section_stats( dd_in[k], idx_sections, fps, **kwargs )

    dim_names = deepcopy(ptdata.names.dim)
    main_label = ptdata.labels.main
    dim_labels = deepcopy(ptdata.labels.dim)
    dimel_labels = deepcopy(ptdata.labels.dimel)
    if cont is False:
        vis ={ 'groupby':None, 'vistype':'cline', 'dlattr':ptdata.vis['dlattr'],
               'sections':False, 'x_ticklabelling':'index' }
        dim_names[-1] = 'section'
        dim_labels[-1] = 'sec.'
        dimel_labels[-1] = 'sec.'
    else: vis = {**ptdata.vis, 'sections':True}

    one_stat = ( isinstance(kwargs['statnames'],str)
                 or (isinstance(kwargs['statnames'],list) and len(kwargs['statnames'])==1) )
    if one_stat:
        main_name = ptdata.names.main + '\n' + "sections' " + kwargs['statnames']
    else:
        main_name = ptdata.names.main + "\nsections' statistics"
        dim_names.insert(axis,'statistics')
        dim_labels.insert(axis,'stats')
        dimel_labels.insert(axis,kwargs['statnames'])

    sextats = PtData(ptdata.topinfo)
    sextats.names.main = main_name
    sextats.names.dim = dim_names
    sextats.labels.main = main_label
    sextats.labels.dim = dim_labels
    sextats.labels.dimel = dimel_labels
    sextats.data = dd_out
    sextats.vis = vis
    sextats.other = ptdata.other.copy()
    return sextats

def corr( ptdata, arr, **kwargs):
    '''
    Correlation between each of several 1-D arrays and one 1-D array.
    Args:
        ptdata (syncoord.ptdata.PtData): The top-level arrays are the first 1-D arrays.
        arr (list,numpy.ndarr): The second 1-D array.
        Optional:
            kind (str): Kind of correlation. Default = 'Kendall' (only currently available).
            sections (bool): True for correlation with sections' means. Default = False
    Returns:
        New PtData object. The last dimension of the arrays has two values: correlation coefficient
            and p-value.
    '''
    kind = kwargs.get('kind','Kendall')
    sections = kwargs.get('sections',False)

    assert kind == 'Kendall', "Only Kendall's rank correlation is currently available."
    corr_lbl = 'Rank'
    dd_in = ptdata.data
    main_name = ptdata.names.main + '\n' + f'{corr_lbl} correlation'
    if sections is True:
        dd_in = secstats( dd_in, statnames='mean', last=True, omitnan=True )
        main_name = main_name + ' with sections'
    dd_out = {}
    for k in dd_in:
        res = stats.kendalltau(dd_in[k], arr, nan_policy='omit')
        dd_out[k] = np.array([res.statistic, res.pvalue])

    main_name = f"{kind}'s corr.({ptdata.names.main}, arr)"
    main_label = deepcopy(ptdata.labels.main)
    dim_labels = deepcopy(ptdata.labels.dim)
    dimel_labels = deepcopy(ptdata.labels.dimel)

    corr = PtData(ptdata.topinfo)
    corr.names.main = main_name
    corr.names.dim = ['coef,p']
    corr.labels.main = ['coef,p']
    corr.labels.dim = ['coef,p']
    corr.labels.dimel = ['coef,p']
    corr.data = dd_out
    corr.vis = {'groupby':None,'printd':True,'dlattr':ptdata.vis['dlattr'],'sections':False,
                   'x_ticklabelling':'index'}
    corr.other = ptdata.other.copy()
    return corr

# .............................................................................
# APPLICATION:

def apply( ptdata, func,*args, verbose=False, **kwargs ):
    '''
    Apply a function to every N-D array of the data dictionary in a PtData object.
    Note: ptdata.dim is copied from the input and may not correspond to the output, except
          for these functions from syncoord.ndarr: tder2D, peaks_to_phase, kuramoto_r, power.
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
        func (Callable): A function to operate on each N-D array of the dictionary.
        Optional:
            axis: Dimension to apply process. Default is -1
            *args, **kwargs: input arguments and keyword-arguments to the function, respectively.
            verbose (bool)
    Returns:
        New PtData object.
    '''
    longname = kwargs.get('longname',-1)
    axis = kwargs.get('axis',-1)
    args_list = list(args)
    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    vis = ptdata.vis.copy()
    dd_in = ptdata.data
    fn = func.__name__

    if fn == 'tder':
        assert 'dim' in kwargs, "missing 1 required keyword argument: 'dim'"
        if kwargs['dim'] > 1:
            del dim_names[axis-1]
            del dim_labels[axis-1]
            del dimel_labels[axis-1]
        main_name = 'Speed'
        main_label = '| $v$ |'
        if ('order' in kwargs) and (kwargs['order'] == 2):
            main_name = 'Absolute Acceleration'
            main_label = '| $a$ |'
    elif fn == 'speed':
        del dim_names[axis-1]
        del dim_labels[axis-1]
        del dimel_labels[axis-1]
        main_name = 'Speed'
        main_label = '| $v$ |'
    elif fn == 'kuramoto_r':
        main_name = 'Kuramoto Order Parameter $r$'
        main_label = '$r$'
        del dim_names[axis-1]
        del dim_labels[axis-1]
        del dimel_labels[axis-1]
        vis = {**vis, 'dlattr':'1.2','vlattr':'r:2f'}
        try:
            if isinstance(dimel_labels[-2],list) or isinstance(dimel_labels[-2],dict):
                vis['vistype'] = 'imshow'
        except: vis['vistype'] = 'line'
    elif fn == 'power':
        main_name = rf'{ptdata.names.main[:].capitalize()} $^{args_list[0]}$'
        main_label = rf'{ptdata.labels.main[:]}$^{args_list[0]}$'
    else:
        if verbose: print('apply - warning: output "dim" field copied from input.')
        main_name = f'{fn}({ptdata.names.main})'
        main_label = fn

    dd_out = {}
    for k in dd_in:
        if fn == 'nanbordz':
            kwargs['fps'] = ptdata.topinfo['fps'].loc[k]
        if (fn in ['speed','tder']) or (fn != 'kuramoto_r'):
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
            for slc,_,midx in ndarr.diter(arr_in,lockdim=[axis-1,axis]):
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

def apply2( ptd_1, ptd_2, func,*args, **kwargs ):
    '''
    Apply a function to corresponding pairs of N-D data arrays in two PtData objects.
    For example, element-wise multiplication of a and b: apply2(a, b, numpy.multiply).
    Args:
        ptd_1, pt_2 (PtData): Data objects with same number of N-D arrays.
                              See documentation for syncoord.ptdata.PtData
        func (Callable): A numpy function to operate on corresponsing N-D arrays.
        Optional:
            *args, **kwargs: input arguments and keyword-arguments to the function, respectively.
    Returns:
        New PtData object.
        The resulting ptdata.names.main will have names.main merged from ptd_1 and ptd_2.
        Other subfileds and the index of the data dict will be of ptd_1.
    '''
    args_list = list(args)
    main_name = f'{func.__name__}({ptd_1.names.main}, {ptd_2.names.main})'
    dim_names = ptd_1.names.dim.copy()
    dim_labels = ptd_1.labels.dim.copy()
    dimel_labels = ptd_1.labels.dimel.copy()
    vis = ptd_1.vis.copy()
    dd_in_1 = ptd_1.data
    dd_in_2 = ptd_2.data

    dd_out = {}
    for k1,k2 in zip(dd_in_1,dd_in_2):
        dd_out[k1] = func(dd_in_1[k1], dd_in_2[k2], *args, **kwargs)

    app2 = PtData(ptd_1.topinfo)
    app2.names.main = main_name
    app2.names.dim = dim_names
    app2.labels.main = ptd_1.labels.main
    app2.labels.dim = dim_labels
    app2.labels.dimel = dimel_labels
    app2.data = dd_out
    app2.vis = vis
    app2.other = deepcopy(ptd_1.other)
    return app2

# .............................................................................
# VISUALISATION:

def visualise( ptdata, **kwargs ):
    # TO-DO: add optional argument 'vistypeopt', which could be a dict with options for each vistype.
    #       For example, if vistype='spectrogram', then vistypeopt['scale']='dB'
    '''
    Visualise data of a PtData object, which normally contains default visualisation information,
    mostly in the 'labels' and 'vis' fields. These may be modified with the optional arguments.
    Args:
        ptdata (PtData): Data object. See documentation for syncoord.ptdata.PtData
        Optional kwargs:
            vistype (str): 'line', 'cline' (circle and line),'spectrogram', or 'imshow'.
            printd (bool): Print data. Not recommended for large data. Default = False
            groupby (int,str,list): N-D array's dimensions to group.
                                    'default' = use defaults: line = 0, spectrogram = -2
            rescale (bool): Rescale visualisation (not data) to min-max of all arrays.
            vscale (float): Vertical scaling.
            hscale (float): Horizontal scaling.
            dlattr (str): Data lines' attributes colour, style, and width (e.g. 'k-0.6')
            sections (bool): Display vertical lines for sections.
            vlattr (str): Vertical lines' attributes colour, style, width, f (full vertical)
                          or b (bits at the top and bottom).
                          Example: 'r:2f' means red, dotted, width=2, full vertical line.
            snumpar (dict): Parameters for section numbrs.
                            snumpar['offset'] (list): Offset factor [horizontal, vertical].
                            snumpar['colour'] (list): RGB. None to not show numbers.
            y_lim (list[float]): Minimum and maximum for vertical axes. Overrides "rescale".
            y_label (str): Label for vertical axis. 'default' uses ptdata.labels.main
                           or 'Hz' if ptdata.names.dim[-2] = 'frequency'
            y_ticks (list[str]): Labels for vertical axis ticks if vistype = 'imshow'
            x_ticklabelling (str): labelling of horizontal axis;
                                   's' = 'time (seconds)',
                                   '25%' = xticks as specified percentage,
                                   '%' = xticks as automatic percentage,
                                   'dim x' = use ptdata.labels.dim[x],
                                   'index','default', or None.
            figtitle (str): Figure's title. If 'default', ptdata.name.main will be used.
            fontsize (float,dict): Float to rescale (default = 1) or dict with keys
                                   'small', 'medium' and 'large' with corresponding values.
            sptitle (bool): True (default) to display subplots' titles. False to not display.
            axes (int): Dimensions to visualise. 1 for 'line' and'spectrogram', 2 for 'imshow'.
            sel_list (list): Selection to display. Also can be input as keywords.
                             See documentation for syncoord.ptdata.select
            retspax (bool): Return subplot axes. Default = False
            savepath (str): Full path (directories, filename, and extension) to save as PNG
    Returns:
        spax (dict{key:list}): Subplot axes, if arg. "retspax" is True. Keys are same as in
                                 topinfo and data.
    '''

    def _xticks_minsec( fps, length_x, vistype, minseps=2 ):
        '''
        Convert and cast xticks expressed in frames to format "minutes:seconds".
        Args:
            fps (float): frame rate
            length_x (float): length of the horizontal axis
            vistype (str): parent function's argument
            Optional:
                miseps (float): minimum separation between the rightmost ticks (seconds)
        '''
        loc_old, _ = plt.xticks()
        if vistype == 'spectrogram':
            specgram_xlims = plt.xlim()
            xlims_diff = specgram_xlims[1] - specgram_xlims[0]
            n_make_new_locs = len(loc_old)-2
            sep_loc_new = xlims_diff / n_make_new_locs
            loc_new = [ specgram_xlims[0].item() + v * sep_loc_new.item()
                        for v in range(n_make_new_locs) ]
            loc_new.append(specgram_xlims[1].item())
            duration = length_x/fps
            lbl_new = [ v - loc_new[0] for v in loc_new]
            lbl_new = [ (v / lbl_new[-1])*duration for v in lbl_new]
            lbl_new = [utils.frames_to_minsec_str(f,1) for f in lbl_new]
        else:
            idx_rem = [ i for i,v in enumerate(loc_old)
                        if ( (v.item() < 0) or (v.item() > length_x)) ]
            loc_new = np.delete(loc_old, idx_rem)
            if loc_new[0] != 0: loc_new = np.insert(loc_new, 0, 0)
            if loc_new[-1] != length_x: loc_new = np.insert(loc_new, len(loc_new), length_x)
            lbl_new = [utils.frames_to_minsec_str(f,fps) for f in loc_new]
            if (lbl_new[-1] == lbl_new[-2]) or ((loc_new[-1] - loc_new[-2]) < (minseps*fps)):
                loc_new = np.delete(loc_new,-2)
                lbl_new = np.delete(lbl_new,-2)
        plt.xticks(loc_new, lbl_new)

    def _xticks_percent( x_percent, length_x, vistype, idx_isochrsec=None ):
        '''
        Make and cast xticks as percentage.
        Args:
            x_percent (int): percentage for ticks or None for automatic
            length_x (float): length of the horizontal axis
            Optional:
                idx_isochrsec (list[int]): index of isochronal sections if x_percent = None
        '''
        if not x_percent:
            if idx_isochrsec: xt = idx_isochrsec
            else: xt, _ = plt.xticks()
            x_percent = 100 * (xt[1]-xt[0]) / length_x
        frac = 100/float(x_percent) # n_sections
        if vistype == 'spectrogram':
            loc_old, _ = plt.xticks()
            specgram_xlims = plt.xlim()
            sep_loc_new = (specgram_xlims[1] - specgram_xlims[0]) / frac
            x_ticks = [ specgram_xlims[0].item() + v * sep_loc_new.item()
                        for v in range(round(frac)) ]
            x_ticks.append( specgram_xlims[1].item() )
            x_labels = [ round(t*x_percent) for t in range(round(frac)+1)]
        else:
            length_section = round(length_x / frac)
            x_ticks = [t*length_section for t in range(round(frac)+1)]
            x_labels = [ round(t*x_percent) for t in range(round(frac)+1)]
            if length_x not in x_ticks: x_ticks[-1] = length_x
        if 100 not in x_labels: x_labels[-1] = 100
        plt.xticks(x_ticks,x_labels)

    def _x_tick_labelling( x_ticklabelling_dictargs ):
        '''
        x_ticklabelling_dictargs.keys:
            vistype, x_ticklabelling, xpercent, fps, idx_isochrsec, hax_len
        '''
        vistype = x_ticklabelling_dictargs['vistype']
        x_ticklabelling = x_ticklabelling_dictargs['x_ticklabelling']
        xpercent = x_ticklabelling_dictargs['xpercent']
        fps = x_ticklabelling_dictargs['fps']
        idx_isochrsec = x_ticklabelling_dictargs['idx_isochrsec']
        hax_len = x_ticklabelling_dictargs['hax_len']
        if x_ticklabelling == 's':
            _xticks_minsec( fps, hax_len, vistype )
        elif x_ticklabelling == 'blank':
            plt.xticks([],[])
        elif x_ticklabelling == 'index':
            plt.xticks((range(hax_len)),([str(v) for v in range(hax_len)]))
        elif xlabel == '%':
            _xticks_percent( xpercent, hax_len, vistype, idx_isochrsec=idx_isochrsec )

    def _overlay_vlines( ax, loc, vlattr, numcolour='k', num_hvoffset=None, numsize=10 ):
        '''
        Overlay vertical lines.
        Args:
            ax (matplotlib.axes): Axis object where to overlay vertical lines.
            loc (list[float]): Location of the lines, in horizontal axis units.
            vlatrr (str): One character for colour, style, width, f (full vertical)
                          or b (bits at the top and bottom).
                          Example: 'r:2f' means red, dotted, width=2, full vertical line.
            Optional:
                numcolor (str): Colour for numbers. None for no numbers.
                num_hvoffset (float): Horizontal and vertical offset for the numbers,
                                      as percentage of axes' lengths.
                numsize (float,int): Font size of the number.
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
            raise Exception( f'Rightmost chracater in "vlattr" \
                             should be "f" or "b", but got "{vlattr[3]}".' )
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
                         horizontalalignment='center', fontsize=numsize)

    kwargs = {**ptdata.vis,**kwargs}
    vistype = kwargs.pop('vistype','line')
    printd = kwargs.pop('printd',False)
    groupby = kwargs.pop('groupby','default')
    rescale = kwargs.pop('rescale',False)
    vscale = kwargs.pop('vscale',1)
    hscale = kwargs.pop('hscale',1)
    dlattr = kwargs.pop('dlattr',None)
    sections = kwargs.pop('sections',True)
    vlattr = kwargs.pop('vlattr','k:2f')
    snumpar = kwargs.pop('snumpar',{})
    snumpar = {'offset':[0,1.13], 'colour':[0.6,0.1,0.2], **snumpar}
    y_lim = kwargs.pop('y_lim',None)
    y_label = kwargs.pop('y_label','default')
    y_ticks = kwargs.pop('y_ticks',None)
    x_ticklabelling = kwargs.pop('x_ticklabelling','s')
    figtitle = kwargs.pop('figtitle','default')
    fontsize = kwargs.pop('fontsize',None)
    sptitle = kwargs.pop('sptitle',True)
    axes = kwargs.pop('axes',-1)
    sel_list = kwargs.pop('sel_list',None)
    retspax = kwargs.pop('retspax',False)
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
    if rescale:
        minmax = [float('inf'),-float('inf')]
        for _,arr in data_dict.items():
            arr_max = np.nanmax(arr)
            arr_min = np.nanmin(arr)
            if arr_min < minmax[0]: minmax[0] = arr_min
            if arr_max > minmax[1]: minmax[1] = arr_max
        minmax = [None if abs(v)==float('inf') else v for v in minmax]
    else: minmax = [None,None]
    if not isinstance(axes,list): axes = [axes]
    try: appaxis_lbl = ptdata.names.dim[axes[-1]]
    except: appaxis_lbl = ''
# TO-DO: case no sections in annotations
    sections_appaxis_exist = f'trimmed_sections_{appaxis_lbl}s' in ptdata.topinfo.columns
    if (appaxis_lbl in kwargs) and sections and sections_appaxis_exist:
        print(f'visualise - warning: sections are disabled for {appaxis_lbl} selection.')
        sections = False
# TO-DO:
    # select time maybe by xlims instead of select_data.
    # sections could follow frame selection.
    if figtitle == 'default': figtitle = ptdata.names.main
    elif (figtitle is None) or (figtitle is False): figtitle = ''
    super_title = ''
    if vistype in ('line','cline'):
        if (groupby == 'default') and (data_dict[data_dict_keys[0]].ndim > len(axes)):
            groupby = [0]
        if (y_lim is None) and rescale: y_lim = minmax
    elif vistype in ('spectrogram','imshow'):
        try:
            if 'spectrogram' in vistype:
                super_title = 'Frequency Spectrum\n'
                ylabel = 'Hz'
            elif ptdata.names.dim[-2] == 'frequency': ylabel = 'Hz'
        except: pass
    else: raise Exception(f"vistype = '{vistype}' is not allowed. Allowed values are \
                            'line', 'cline','spectrogram', and 'imshow'")
    if groupby == 'default':
        if 'imshow' in vistype: groupby = -2
        else: groupby = None
    spt_y = 1
    if sections and sections_appaxis_exist:
        spt_y = snumpar['offset'][1]*1.1
        xticks_percent__sections = True
    else: xticks_percent__sections = False
    data_shape = list(data_dict[data_dict_keys[0]].shape)
    sing_dims = False
    i_1 = []
    if 1 in data_shape: # singleton dimensions
        i_1 = [i for i, x in enumerate(data_shape) if x == 1]
        if not isinstance(groupby,list): groupby = [groupby]
        groupby.extend(i_1)
        sing_dims = True
    if groupby is None: groupby = axes.copy()
    if not isinstance(groupby,list): groupby = [groupby]
    groupby.extend(axes)
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
    if (x_ticklabelling is None) or ('%' not in x_ticklabelling):
        if figtitle.endswith('isochronal sections)'):
            x_ticklabelling = default_xtick_percentage_str
    xpercent = None

    xlabel = None
    if x_ticklabelling is None:
        xlabel = ''
        x_ticklabelling = 'blank'
    elif x_ticklabelling == 's':
        if 'sections statistics' in figtitle:
            x_ticklabelling = 'blank'
            xlabel = None
        else: xlabel = 'time (m:s)'
    elif 'dim' in x_ticklabelling:
        xlabel = ptdata.labels.dim[int(x_ticklabelling.split(' ')[1])]
    elif '%' in x_ticklabelling:
        xlabel = '%'
        xpercent = x_ticklabelling.replace('%','')
        if xpercent: xpercent = float(xpercent)
    n_sel_top = len(data_dict)
    n_sp = int(np.prod(s)*n_sel_top)
    n_title_lines = ptdata.names.main.count('\n')+1
    fig_height = (n_sp * 2.4 + n_title_lines*0.2 )*vscale
    i_sp = 1
    fig = plt.figure(figsize=(12*hscale,fig_height))
    if y_ticks is not None:
        sp_yticks = []
    idx_isochrsec = None
    x_ticklabelling_dictargs = dict( vistype=vistype, x_ticklabelling=x_ticklabelling,
                                     xpercent=xpercent )
    font_sizes = {'small':10, 'medium': 12, 'large': 16}
    if isinstance(fontsize,(float,int)): font_sizes = {k:v*fontsize for k,v in font_sizes.items() }
    elif isinstance(fontsize,dict): font_sizes = {**font_sizes, **fontsize}
    elif fontsize is not None: raise Exception('Invalid value for arg "fontsize".')
    spax = {}
    for i_top in range(n_sel_top):
        fps = ptdata.topinfo['fps'].iloc[i_top]
        k = data_dict_keys[i_top]
        top_arr = data_dict[k]
        new_i_top = True
        if xticks_percent__sections:
            idx_isochrsec = ptdata.topinfo[f'trimmed_sections_{appaxis_lbl}s'].loc[k]
        x_ticklabelling_dictargs['fps'] = fps
        x_ticklabelling_dictargs['idx_isochrsec'] = idx_isochrsec
        array_iterator = ndarr.diter( top_arr, lockdim=groupby )
        spax[k] = []
        for vis_arr,i_ch,i_nd in array_iterator:
# TO-DO: Arg. 'topidxkey' (boolean) to add index and key of the top array to the title.
            sp_title = ''
            if new_i_top:
                if 'Name' in ptdata.topinfo.columns:
                    sp_title = '"'+ptdata.topinfo['Name'].iloc[i_top]+'"'
                elif 'Group' in ptdata.topinfo:
                    sp_title = 'Group: "'+ptdata.topinfo['Group'].iloc[i_top]+'"'
            ax = plt.subplot(n_sp, 1, i_sp)
            spax[k].append(ax)
            if sing_dims: vis_arr = np.squeeze(vis_arr)
            hax_len = vis_arr.shape[axes[-1]]
            x_ticklabelling_dictargs['hax_len'] = hax_len
            if vistype in ('line','cline','imshow'):
                if vistype in ('line','cline'):
                    dictargs_plot = {}
                    if vistype == 'cline':
                        dictargs_plot['marker'] = 'o'
                        dictargs_plot['markersize'] = dlattr_[2]*5
                    plt.plot( vis_arr.T, color=dlattr_[0], linestyle=dlattr_[1],
                              linewidth=dlattr_[2], **dictargs_plot )
                    if x_ticklabelling != 'index': plt.xlim((0,hax_len))
                elif vistype == 'imshow':
                    if vis_arr.ndim != 2:
                        check_gb = abs(2-vis_arr.ndim)
                        exmsg = ''.join([f'Number of dimensions for imshow is {vis_arr.ndim} ',
                                          'but should be 2. Check that argument "groupby" has ',
                                         f'length = {check_gb}, or use another "vistype" value'])
                        raise Exception(exmsg)
                    isextent = [0, hax_len, 0, vis_arr.shape[0]]
                    plt.imshow( vis_arr, origin='lower', aspect='auto', extent=isextent,
                                vmin=minmax[0], vmax=minmax[1] )
            elif 'spectrogram' in vistype:
                plt.specgram(vis_arr,Fs=fps,detrend='linear',scale='linear')
            if y_lim: plt.ylim(y_lim)
            if (y_ticks is not None) and (vis_arr.ndim == 2) and (vistype == 'imshow'):
                if isinstance(y_ticks,list): sp_yticks.append(y_ticks)
                elif isinstance(y_ticks,dict): sp_yticks.append(y_ticks[ k ])
            plt.ylabel(ylabel, fontsize=font_sizes['small'])
            _x_tick_labelling( x_ticklabelling_dictargs )
            plt.xticks(fontsize=font_sizes['small'])
            plt.yticks(fontsize=font_sizes['small'])
            if sections and sections_appaxis_exist:
                vlsec = ptdata.topinfo[f'trimmed_sections_{appaxis_lbl}s'].iloc[i_top]
                if 'spectrogram' in vistype:
                    xstart,xend = plt.xlim()
                    vlsec = [ ( (v/hax_len)*(xend-xstart)+xstart ).item() for v in vlsec]
                    vlattr = vlattr.replace('k','w')
                _overlay_vlines( plt.gca(), vlsec, vlattr, numcolour=snumpar['colour'],
                                 num_hvoffset=snumpar['offset'], numsize=font_sizes['small'] )
            for i in i_ch:
                if isinstance(ptdata.labels.dimel[i],dict): # dict: different labels for each top array
                    sp_lbl = ptdata.labels.dimel[i][k][i_nd[i]]
                elif isinstance(ptdata.labels.dimel[i],list):
                    sp_lbl = ptdata.labels.dimel[i][i_nd[i]] # list: same labels for all top arrays
                else:
                    sp_lbl = ptdata.labels.dimel[i] # (singleton dim.) same labels for all top arrays
                sp_title = ''.join([sp_title,'\n',sp_lbl])
            if printd: print(np.round(vis_arr,3))
            if sptitle and sp_title:
                plt.title(sp_title,y=spt_y,fontsize=font_sizes['medium'])
            if new_i_top: new_i_top = False
            i_sp += 1
    fig.supxlabel(xlabel,fontsize=font_sizes['medium'])
    plt.suptitle( super_title + figtitle, fontsize=font_sizes['large'] )
    plt.tight_layout(rect=[0, 0.005, 1, 0.98])
    # TO-DO: this might leave a bit too much space in between ticks:
    if (y_ticks is not None) and (vis_arr.ndim == 2):
        for k in data_dict_keys:
            for i_ax, ax_ in enumerate(spax[k]):
                yticks_loc = ax_.get_yticks()
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
                spax[k][i_ax].set_yticks(yticks_loc,labels=yticks_lbl)
    if savepath: fig.savefig(savepath + '.png')
    if retspax is True:  return spax
    plt.pause(0.1)
