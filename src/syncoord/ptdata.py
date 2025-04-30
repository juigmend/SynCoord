'''Data class PtData and functions that have ptdata objects as input.'''

from copy import deepcopy

import numpy as np
from scipy import signal
from scipy.fft import fftfreq
import matplotlib.pyplot as plt

from . import ndarr, utils

# .............................................................................
# DATA HANDLING:

class SubField:
    '''
    Empty class used to add sub-fields to an object's attribute.
    '''
    pass

class PtData:
    '''
    Data-oriented class.
    Attributes:
        names:
            names.main: str, main name of the object. Descriptive.
                        It may grow as the names of processes to the data are added.
            names.dim: list of str, dimensions of the object. Concise, used for selection.
        labels:
            labels.main: str, main label of the object. Concise or abbreviated.
            labels.dim: list of str, dimensions of the object. Concise or abbreviated.
                         Used for visualisations.
            labels.dimel: list of str, elements of dimensions. It can also be a dict
                          of lists with one item per data array, if the arrays are inhomogeneous.
                          Used for visualisations.
        data: dict where each item should be an N-D data array (type numpy.ndarray) containig data.
              Each item is at the top level of the data hierarchy. Its order and index should
              directly correspond to the index of topinfo.
        topinfo: Pandas dataframe with information for each entry at the top level.
                 See documentation for syncoord.utils.load_data
        vis: dict of keyword arguments for default visualisation parameters,
             when the ptdata object is passsed to ptdata.visualise
        other: empty dict, for any other information.
    Methods:
        print_shape(): prints shape of data arrays.
        print(): prints attributes and shape of data arrays
        checkdim(verbose=True): checks consistency of dimensions of data arrays (shape),
                                except last dimension.
                                Returns -1 if empty, 0 if inconsistent, 1 if consistent.
        visualise(**kwargs): calls syncoord.ptdata.visualise with the same optional arguments.
    Note:
        Use command vars to see content of subfields.
        Examples:
            ptdata = PtData(topinfo)   # initialise data object
            ptdata.names.main = 'Juan' # assign main name
            vars(ptdata.names)
            returns: {'main': 'Juan'}
    Args:
        topinfo: See documentation for syncoord.utils.load_data
    '''
    def __init__(self,topinfo):
        self.names = SubField()
        self.names.main = ''
        self.names.dim = []
        self.labels = SubField()
        self.labels.main = ''
        self.labels.dim = []
        self.labels.dimel = []
        self.data = {}
        self.topinfo = topinfo
        self.vis = {}
        self.other = {}

    def print_shape(self):
        print('data:')
        for k in self.data: print(f'key = {k}, shape = {self.data[k].shape}')
        print()

    def print(self):
        print(f'names:\n{vars(self.names)}\n')
        print(f'labels:\n{vars(self.labels)}\n')
        if self.data: self.print_shape()
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
                            print('Inconsistent array dimensions (except last dimension).')
                            self.print_shape()
                        return 0
                return 1
        else:
            if verbose: print('Field "data" is empty.')
            return -1

    def visualise(self,**kwargs):
        visualise(self,**kwargs)

def load( preproc_data, *prop_path, annot_path=None, max_n_files=None,
              print_info=True, **kwargs ):
    '''
    Args:
        preproc_data: str, dict or np.ndarray
                      If str: folder with parquet files for preprocesed data
                              (e.g., r"~/preprocessed"),
                              or "make" to produce synthetic data with default values.
                      If dict: as returned by syncoord.utils.testdata
                      If np.ndarray: as returned by syncoord.utils.init_testdatavars
        prop_path: str, path and filename for properties CSV file (e.g., r"~/properties.csv").
                   Optional or ignored if preproc_data = "make"
        Optional:
            annot_path: str, path and filename of annotations CSV file
                        (e.g., r"~/String_Quartet_annot.csv").
            max_n_files: None (all) or scalar, number of files to load.
            print_info: bool, print durations of data.
            **kwargs: passed to syncoord.utils.load_data
                      and syncoord.utils.init_testdatavars if preproc_data = "make"
    Returns:
        PtData object with loaded data (dictionary of mutlti dimensional numpy arrays).
    '''
    load_out = utils.load_data( preproc_data, prop_path, annot_path=annot_path,
                                max_n_files=max_n_files, print_info=print_info, **kwargs )

    pos = PtData(load_out[0])
    pos.names.main = 'Position'
    pos.names.dim = load_out[1]
    pos.labels.main = 'Pos.'
    pos.labels.dim = load_out[2]
    pos.labels.dimel = load_out[3]
    pos.data = load_out[4]
    pos.vis['y_label'] = None
    pos.vis['dlattr'] = '1.2'
    return pos

def select( ptdata,*args,**kwargs ):
    '''
    Args:
        ptdata: PtData object
        Either of these:
            * Location for top-level and dimensions,
              as a list ordered as [top-level, dim 1, dim 2, ...].
            * Arbitrary number of keywords and values, separated or in a dictionary.
              Keywords are 'top' and dimension names.
            In both cases values are locations or index. They may be 'all', int,
            or slice(start,end), separated or in a nested list with non-continguous values.
    Returns:
        sel: PtData object containing the selected data and topinfo.
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

    def checktype_(obj):
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
        for k in name_sel: checktype_(name_sel[k])
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
        for el in loc_sel: checktype_(el)
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

def gensec( ptdata, n, print_info=False ):
    '''
    Generate equally spaced sections for each data array (along the last dimension).
    The object will be updated in place.
    Args:
        ptdata: a ptdata object. A prompt will ask to replace the following columns
                of ptdata.topinfo if they exist: ptdata.topinfo['Sections'];
                                                 ptdata.topinfo['trimmed_sections_frames']
        n: number of equally spaced sections, conforming to rounding precision.
        Optional:
            print_info: Boolean. True will print sections' length and difference in frames.
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
            print('Warning: function exited without updating ptdata.topinfo')
            return
    all_equal_sex_f = []
    all_equal_sex_s = []
    d_keys = list(ptdata.data.keys())
    info_title = True
    for i,k in enumerate(ptdata.topinfo.index):
        if k != d_keys[i]:
            raise Exception("".join([f"ptdata.topinfo.index[{k}] doesn't match ",
                                     f"list(ptdata.data.keys())[{i}]"]))
        length_sec = ptdata.data[k].shape[-1]/n
        equal_sex_f = [round(i*length_sec) for i in range(1,n)]
        all_equal_sex_f.append(equal_sex_f)
        fps = ptdata.topinfo.loc[k,'fps']
        equal_sex_s = [v/fps for v in equal_sex_f]
        all_equal_sex_s.append(equal_sex_s)
        if print_info:
            if info_title:
                print("key; sections' length (frames); difference (frames):")
                info_title = False
            ss = [0] + equal_sex_f + [ptdata.data[k].shape[-1]]
            sl = [ ss[i+1]-ss[i] for i in range(len(ss)-1) ]
            d = [ sl[i+1]-sl[i] for i in range(len(sl)-1) ]
            print(f'  {k};  {sl};  {d}')
    # ptdata.topinfo['Sections'] = all_equal_sex_s
    ptdata.topinfo['trimmed_sections_frames'] = all_equal_sex_f

# .............................................................................
# ANALYSIS-ORIENTED OPERATIONS:

def smooth( ptdata,**kwargs ):
    '''
    Apply filter to ptdata, row-wise, to a dimension of N-D arrays (default is last dimension).
    Args:
        ptdata: PtData object. See documentation for syncoord.ptdata.PtData
        filter_type: 'butter' or 'mean'
        Options:
            axis: int or str, default = -1.
                  Note: axis is a dimension of the N-D array.
                        The rightmost axis (-1) is the fastest changing.
            If filter_type = 'butter':
                freq_response: 'lowpass' (LPF),'highpass' (HPF), or 'bandpass' (BPF).
                cutoff_freq (float or list): cutoff (LPF and HPF) or center frequency(BPF) (Hz).
                order (int)
                bandwidth (float): only for 'bandpass' (Hz).
            If filter_type = 'mean':
                window_size (float or list): (seconds)
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
    vis_opt = {**ptdata.vis,'dlattr':'1.2'}
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

def tder2D( ptdata, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.tder2D
    '''
    return apply( ptdata, ndarr.tder2D, **kwargs )

def peaks_to_phase( ptdata, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.peaks_to_phase
    '''
    return apply( ptdata, ndarr.peaks_to_phase, **kwargs )

def kuramoto_r( ptdata ):
    '''
    Wrapper for syncoord.ndarr.kuramoto_r
    '''
    return apply( ptdata, ndarr.kuramoto_r )

def fourier( ptdata, window_duration, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.fourier_transform
    Args:
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
        window_duration: length of the FFT window in seconds unless optional parameter fps = None.
        Optional:
            **kwargs: input parameters to the fourier_transform function.
                      See documentation for syncoord.ndarr.fourier_transform
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
        if rdif < 0.01: freq_bins_rounded = np.round(freq_bins_rounded,0).astype(int)
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
        vis['y_ticks'] = freq_bins
    dim_names.insert(-1,'frequency')
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

def winplv( ptdata, window_duration, window_hop=None, pairs_axis=0,
            fixed_axes=None, plv_axis=-1, mode='same', verbose=False ):
    '''
    Pairwise sliding-window Phase-Locking Values.
    Args:
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
                N-D arrays should have at least 2 dimensions.
        window_duration: window length in seconds
        Optional:
            window_hop: window step in seconds. None for a step of 1 frame.
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
                Note: axis is a dimension of the N-D array.
                      The rightmost axis (-1) is the fastest changing.
            mode: 'same' (post-process zero-padded, same size as input)
                   or 'valid' (result of windowed process).
                   Note: If mode='valid', the sections (ptdata.topinfo['Sections'] and
                   ptdata.topinfo['trimmed_sections_frames']) will be shifted accordingly.
            verbose: Display progress.
    Returns:
        New PtData object.
    '''
    if not window_hop: window_step = 1
    else:
        window_step = None
        new_fps = []
    dd_in = ptdata.data
    dd_out = {}
    c = 1
    for k in dd_in:
        if verbose:
            print(f'processing array {k} ({c} of {len(ptdata.data.keys())})')
            c+=1
        fps = ptdata.topinfo.loc[k,'fps']
        window_length = round(window_duration * fps)
        if window_hop:
            window_step = round(window_hop * fps)
            new_fps.append(fps/window_step)
        if fixed_axes is None:
            if dd_in[k].ndim > 2:
                fixed_axes = [-2,-1]
            elif dd_in[k].ndim == 2:
                fixed_axes = -1
            if dd_in[k].ndim < 2:
                raise Exception('number of dimensions in data arrays should be at least 2')
        dd_out[k], pairs_idx = ndarr.apply_to_pairs( dd_in[k], ndarr.windowed_plv,
                                                     pairs_axis, fixed_axes=fixed_axes,
                                                     window_length=window_length, mode=mode,
                                                     window_step=window_step,
                                                     axis=plv_axis, verbose=verbose  )

    if mode == 'valid':
        topinfo = utils.trim_topinfo_start(ptdata,window_duration/2)
    else:
        topinfo = ptdata.topinfo
    if window_hop:
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

    wplv = PtData(topinfo)
    wplv.names.main = 'Pairwise Windowed Phase-Locking Value'
    wplv.names.dim = dim_names
    wplv.labels.main = 'PLV'
    wplv.labels.dim = dim_labels
    wplv.labels.dimel = dimel_labels
    wplv.data = dd_out
    wplv.vis = {'dlattr':'1.2','groupby':groupby, 'vistype':vistype, 'vlattr':'r:3f'}
    if 'freq_bins' in ptdata.other:
        wplv.vis['y_ticks'] = ptdata.other['freq_bins'].copy()
    wplv.other = ptdata.other.copy()
    return wplv

def xwt( ptdata, minmaxf, pairs_axis, fixed_axes, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.xwt_nd
    Pairwise multi-dimensional cross-wavelet spectrum.
    Args:
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
                N-D arrays should have at least 2 dimensions.
        minmaxf: list with minimum and maximum frequency (Hz).
        pairs_axis: axis to form the pairs.
        fixed_axes: axis (int) or axes (list) that are passed to the xwt_nd function.
        Optional:
            Keyword arguments to syncoord.ndarr.xwt_nd
            verbose: bool, it will apply to syncoord.ndarr.apply_to_pairs and syncoord.ndarr.xwt_nd
    Returns:
        New PtData object.
    '''
    verbose = kwargs.get('verbose',True)
    if 'matlabeng' in kwargs: neweng = False
    else:
        neweng = True
        genxwt_path = kwargs.pop('gxwt_path',None)
        xwtnd_path = kwargs.pop('xwtnd_path',None)
        addpaths = [genxwt_path,xwtnd_path]
        kwargs['matlabeng'] = utils.matlab_eng(addpaths,verbose)

    dd_in = ptdata.data
    dd_out = {}
    c = 1
    for k in dd_in:
        arr_nd = dd_in[k]
        if arr_nd.ndim < 3: raise Exception(f'Data dimensions should be at least 2,\
                                              but currently are {arr_nd.ndim}')
        fps = ptdata.topinfo.loc[k,'fps']
        pairs_results = ndarr.apply_to_pairs( arr_nd, ndarr.xwt_nd, pairs_axis,
                                              fixed_axes=fixed_axes,
                                              minmaxf=minmaxf, fps=fps, **kwargs )
        dd_out[k] = pairs_results[0]

    pairs_idx = pairs_results[1]
    freq_bins = pairs_results[2][0][0].tolist()
    freq_bins_round = np.round(freq_bins,1).tolist()
    if neweng:
        kwargs['matlabeng'].quit()
        if verbose: print('Disconnected from Matlab.')

    dim_names = ptdata.names.dim.copy()
    dim_labels = ptdata.labels.dim.copy()
    dimel_labels = ptdata.labels.dimel.copy()
    if isinstance(fixed_axes,list): groupby = fixed_axes[0]
    else: groupby = fixed_axes
    i_freq_lbl = groupby
    if i_freq_lbl < 0: i_freq_lbl = len(dim_names) + i_freq_lbl
    if (i_freq_lbl != pairs_axis) and (i_freq_lbl >= 0):
        dim_names[i_freq_lbl] = 'frequency'
        dim_labels[i_freq_lbl] = 'freq.'
        dimel_labels[i_freq_lbl] = freq_bins_round
    dim_names[pairs_axis] = 'pair'
    dim_labels[pairs_axis] = 'pairs'
    k = list(dd_in.keys())[0]
    n_pair_el = dd_in[k].shape[pairs_axis]
    n_pairs = (n_pair_el**2 - n_pair_el)//2
    dimel_labels[pairs_axis] = ['pair '+str(p) for p in pairs_idx]
    if groupby == -1: vistype = 'line'
    else: vistype = 'imshow'

    xwtdata = PtData( deepcopy(ptdata.topinfo) )
    xwtdata.names.main = 'Cross-Wavelet Spectrum'
    xwtdata.names.dim = dim_names
    xwtdata.labels.main = 'XWS'
    xwtdata.labels.dim = dim_labels
    xwtdata.labels.dimel = dimel_labels
    xwtdata.data = dd_out
    xwtdata.vis = { 'groupby':groupby, 'vistype':vistype, 'rescale':True,
                    'dlattr':'1.2', 'vlattr':'r:3f', 'y_ticks':freq_bins_round }
    xwtdata.other['freq_bins'] = freq_bins
    return xwtdata

def rho( ptdata, exaxes=None, mode='all' ):
    '''
    Group synchrony.
    Wrapper for multiSyncPy.synchrony_metrics.rho
    Args:
        ptdata: PtData object with phase angles.
                See documentation for syncoord.ptdata.PtData
        exaxes: int or None.
                If int: dimension(s) to exclude from grouping, except last dimension.
                If None: all dimensions except last will be grouped.
                Set to -2 if that dimension's name is 'frequency'.
        Optional:
            mode: 'all' or 'mean'
    Returns:
        New PtData object.
    References:
        https://doi.org/10.3389/fphys.2012.00405
        https://github.com/cslab-hub/multiSyncPy
    '''
    if len(ptdata.data)==1: cdv = False
    else: cdv = True
    assert ptdata.checkdim(verbose=cdv)==1
    from multiSyncPy.synchrony_metrics import rho as sm_rho

    if mode == 'all':
        i_out = 0
        n_out = mode
    elif mode == 'mean': i_out = n_out = 1
    else: raise Exception('mode can only be "all" or "mean"')

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
    ptd_rho.names.main = 'Rho'
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
        ptdata: PtData object with topinfo containing column 'trimmed_sections_frames'.
                See documentation for syncoord.utils.load_data
        Optional:
            last: If True, from the last index of sections to the end will be the last section.
            axis: axis to apply the process.
                  Note: axis is a dimension of the N-D array.
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
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
                The field topinfo should have columns 'trimmed_sections_frames',
                with lists having equally spaced indices.
                See documentation for syncoord.ptdata.isochrsec
        Optional:
            aggregate_axes: scalar or list indicating the axes to aggregate.
            sections_axis: the axes of the aggregated axes where the sections are taken from.
                  For example:
                      data.shape = (6,1,15,9000)
                      aggregate_axes = [-2,-1]:
                          shape of aggregated axes: (15,9000)
                      sections_axis = 1:
                          The sections are taken from the axis with length 9000 points.
                Note: axis is a dimension of the N-D array.
                      The rightmost axis (-1) is the fastest changing.
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
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
        Optional:
            function: 'sum' or 'mean'
            axis to run the operation
                Note: axis is a dimension of the N-D array.
                      The rightmost axis (-1) is the fastest changing.
    Returns:
        New PtData object.
    '''
    dd_in = ptdata.data
    dd_out = {}
    for k in dd_in:
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
    vis = {**ptdata.vis, 'groupby':axis, 'sections':True}
    other =  deepcopy(ptdata.other)
    if 'frequency' not in ptdata.names.dim:
        if 'y_ticks' in vis: del vis['y_ticks']
        if 'freq_bins' in other: del other['freq_bins']

    agg = PtData(ptdata.topinfo)
    agg.names.main = main_name
    agg.names.dim = dim_names
    agg.labels.main = ptdata.labels.main
    agg.labels.dim = dim_labels
    agg.labels.dimel = dim_labels
    agg.data = dd_out
    agg.vis = vis
    agg.other = other
    return agg

def aggrtop( ptdata, function='mean' ):
    '''
    Wrapper for numpy.sum or numpy.mean
    Aggregate top-level homogeneous N-D data arrays.
    Args:
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
        Optional:
            function: 'sum' or 'mean'
    Returns:
        New PtData object. Property 'topinfo' will retain only columns whose rows are identical,
        collapsed into a single row.
    '''
    dd_in = ptdata.data
    arr_nd_out = np.zeros(dd_in[next(iter(dd_in))].shape)
    for k in dd_in: arr_nd_out += dd_in[k]
    if function == 'mean': arr_nd_out = arr_nd_out/len(dd_in)
    dd_out = {0:arr_nd_out}

    main_name = ptdata.names.main
    if function == 'sum': function_lbl = 'Added'
    elif function == 'mean': function_lbl = 'Mean'
    aggr_lbl = f'\n({function_lbl} top-level data)'
    colnames_identical = []
    for colname in ptdata.topinfo:
        if len(ptdata.topinfo[colname].drop_duplicates()) == 1:
            colnames_identical.append(colname)
    topinfo = ptdata.topinfo[colnames_identical].iloc[0].copy().to_frame().T.reset_index(drop=True)

    agg = PtData(topinfo)
    agg.names.main = ptdata.names.main + aggr_lbl
    agg.names.dim = ptdata.names.dim.copy()
    agg.labels.main = ptdata.labels.main
    agg.labels.dim = ptdata.labels.dim.copy()
    agg.labels.dimel = ptdata.labels.dimel.copy()
    agg.data = dd_out
    agg.vis = {**ptdata.vis, 'groupby':0, 'sections':True}
    agg.other = deepcopy(ptdata.other)
    return agg

def secstats( ptdata, **kwargs ):
    '''
    Wrapper for syncoord.ndarr.section_stats
    Descriptive statistics for sections of N-D data arrays.
    Args:
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
        Optional:
            last: If True, from the last index of sections to the end will be the last section.
            margins: scalar, list, or dict. Trim at the beginning and ending, in seconds.
                     If scalar: same trim bor beginning and ending.
                     If list: trims for beginning and ending. Nested lists for sections.
                     If dict: items correspond to N-D data arrays, keys are same as ptdata.data
                              (and as in ptdata.topinfo), and values are lists.
            axis to run the process.
            statnames: str or list of statistics to compute. Default is all.
                       Available stats: 'mean','median','min','max','std'.
    Returns:
        New PtData object.
    '''
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

    main_name = ptdata.names.main + '\nsections statistics'
    dim_names = deepcopy(ptdata.names.dim)
    dim_names.insert(axis,'statistics')
    main_label = ptdata.labels.main
    dim_labels = deepcopy(ptdata.labels.dim)
    dim_labels.insert(axis,'stats')
    dimel_labels = deepcopy(ptdata.labels.dimel)
    dimel_labels.insert(axis,kwargs['statnames'])

    sextats = PtData(ptdata.topinfo)
    sextats.names.main = main_name
    sextats.names.dim = dim_names
    sextats.labels.main = main_label
    sextats.labels.dim = dim_labels
    sextats.labels.dimel = dimel_labels
    sextats.data = dd_out
    sextats.vis = {'groupby':None,'vistype':'cline','dlattr':ptdata.vis['dlattr'],'sections':False}
    sextats.other = ptdata.other.copy()
    return sextats

# .............................................................................
# APPLICATION:

def apply( ptdata, func,*args, **kwargs ):
    '''
    Apply a function to every N-D array of the data dictionary in a PtData object.
    Note: ptdata.dim is copied from the input and may not correspond to the output, except
          for these functions from syncoord.ndarr: tder2D, peaks_to_phase, kuramoto_r, power.
    Args:
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
        func: a function to operate on each N-D array of the dictionary.
        Optional:
            axis: dimension to apply process. Default is -1.
            *args, **kwargs: input arguments and keyword-arguments to the function, respectively.
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
        vis = {**vis, 'dlattr':'1.2','vlattr':'r:2f'}
        if isinstance(dimel_labels[-2],list) or isinstance(dimel_labels[-2],dict):
            vis['vistype'] = 'imshow'
    elif fn == 'power':
        main_name = rf'{ptdata.names.main[:].capitalize()} $^{args_list[0]}$'
        main_label = rf'{ptdata.labels.main[:]}$^{args_list[0]}$'
    else:
        print('Warning: output "dim" field copied from input.')
        main_name = f'{fn}({ptdata.names.main})'
        main_label = fn

    dd_out = {}
    for k in dd_in:
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
        ptd_1, pt_2: PtData objects, see documentation for syncoord.ptdata.PtData
        func: a numpy function to operate on corresponsing N-D arrays.
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
    '''
    Visualise data of a PtData object, which normally contains default visualisation information,
    mostly in the 'labels' and 'vis' fields. These may be modified with the optional arguments.
    Args:
        ptdata: PtData object, see documentation for syncoord.ptdata.PtData
        Optional:
            vistype: 'line', 'cline' (circle and line),'spectrogram', or 'imshow'
            groupby: int, str, or list, indicating N-D array's dimensions to group.
                     'default' = use defaults: line = 0, spectrogram = -2
            rescale: bool, rescale visualisation (not data) to min-max of all arrays.
            vscale: float, vertical scaling.
            dlattr: str, data lines' attributes colour, style, and width (e.g. 'k-0.6')
            sections: display vertical lines for sections. True or False.
            vlattr: vertical lines' attributes. str with one character for
                    colour, style, width, f (full vertical) or b (bits at the top and bottom).
                    For example: 'r:2f' means red, dotted, width=2, full vertical line.
            snum_hvoff: list with horizontal and vertical offset factor for section numbers.
            y_lim: list with minimum and maximum for vertical axes. Overrides arg. 'rescale'.
            y_label: label for vertical axis. 'default' uses ptdata.labels.main
                     or 'Hz' if ptdata.names.dim[-2] = 'frequency'
            y_ticks: labels for vertical axis ticks, useful only when vistype = 'imshow'
            x_ticklabelling: labelling of horizontal axis;
                             's' = 'time (seconds)',
                             '25%' = xticks as specified percentage,
                             '%' = xticks as automatic percentage,
                             'dim x' = use ptdata.labels.dim[x], or None.
            figtitle: figure title. If None, ptdata.name.main will be used.
            axes: dimensions to visualise. One for 'line' and'spectrogram', two for 'imshow'.
            sel_list: selection to display with list *
            savepath: full path (directories and filename with extension) to save as PNG
            more **kwargs: selection to display, passed to syncoord.ptdata.select_data
    '''
    def xticks_minsec_( fps, length_x, vistype, minseps=2 ):
        '''
        Convert and cast xticks expressed in frames to format "minutes:seconds".
        Args:
            fps: frame-rate.
            length_x: size of the horizontal axis.
            vistype: parent function's argument
            Optional:
                miseps: minimum separation between the rightmost ticks (seconds)
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

    def xticks_percent_( x_percent, length_x, vistype, idx_isochrsec=None ):
        '''
        Make and cast xticks as percentage.
        Args:
            x_percent: (int) percentage for ticks or None for automatic.
            length_x: size of the horizontal axis.
            Optional:
                idx_isochrsec: index of isochronal sections, if x_percent = None.
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

    def x_tick_labelling_( x_ticklabelling_dictargs ):
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
            xticks_minsec_( fps, hax_len, vistype )
        elif x_ticklabelling == 'blank':
            plt.xticks([],[])
        elif xlabel == '%':
            xticks_percent_( xpercent, hax_len, vistype, idx_isochrsec=idx_isochrsec )

    def overlay_vlines_( ax, loc, vlattr, numcolour='k', num_hvoffset=None ):
        '''
        Args:
            ax: pyplot axis object where to overlay vertical lines.
            loc: list with the location of the lines, in horizontal axis units.
            vlatrr: str with one character for each of these:
                    colour, style, width, f (full vertical) or b (bits at the top and bottom).
                    For example: 'r:2f' means red, dotted, width=2, full vertical line.
            Optional:
                numcolor: Colour for numbers. None for no numbers.
                num_hvoffset: horizontal and vertical offset for the numbers,
                              as percentage of axes' lengths.
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
                         horizontalalignment='center')

    kwargs = {**ptdata.vis,**kwargs}
    vistype = kwargs.pop('vistype','line')
    groupby = kwargs.pop('groupby','default')
    rescale = kwargs.pop('rescale',False)
    vscale = kwargs.pop('vscale',1)
    dlattr = kwargs.pop('dlattr',None)
    sections = kwargs.pop('sections',True)
    vlattr = kwargs.pop('vlattr','k:2f')
    snum_hvoff = kwargs.pop('snum_hvoff',[0,1.13])
    y_lim = kwargs.pop('y_lim',None)
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
    if rescale:
        minmax = [float('inf'),-float('inf')]
        for _,arr in data_dict.items():
            arr_max = arr.max()
            arr_min = arr.min()
            if arr_min < minmax[0]: minmax[0] = arr_min
            if arr_max > minmax[1]: minmax[1] = arr_max
    else: minmax = [None,None]
    if not isinstance(axes,list): axes = [axes]
    appaxis_lbl = ptdata.names.dim[axes[-1]]
# TO-DO: case no sections in annotations
    sections_appaxis_exist = f'trimmed_sections_{appaxis_lbl}s' in ptdata.topinfo.columns
    if (appaxis_lbl in kwargs) and sections and sections_appaxis_exist:
        print(f'Warning: sections are disabled for {appaxis_lbl} selection.')
        sections = False
# TO-DO:
    # select time maybe by xlims instead of select_data.
    # sections could follow frame selection.
    if figtitle is None: figtitle = ptdata.names.main
    super_title = ''
    if vistype in ('line','cline'):
        if (groupby == 'default') and (data_dict[data_dict_keys[0]].ndim > len(axes)):
            groupby = [0]
        if (y_lim is None) and rescale: y_lim = minmax
    elif vistype in ('spectrogram','imshow'):
        if 'spectrogram' in vistype:
            super_title = 'Frequency Spectrum\n'
            ylabel = 'Hz'
        elif ptdata.names.dim[-2] == 'frequency': ylabel = 'Hz'
    else: raise Exception(f"vistype = '{vistype}' is not allowed. Allowed values are \
                            'line', 'cline','spectrogram', and 'imshow'")
    if groupby == 'default':
        if 'imshow' in vistype: groupby = -2
        else: groupby = None
    spt_y = 1
    if sections and sections_appaxis_exist:
        spt_y = snum_hvoff[1]*1.1
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
    fig = plt.figure(figsize=(12,fig_height))
    if y_ticks is not None:
        sp_yticks = []
        sp_axes = []
    idx_isochrsec = None
    x_ticklabelling_dictargs = dict( vistype=vistype, x_ticklabelling=x_ticklabelling,
                                     xpercent=xpercent )
    for i_top in range(n_sel_top):
        fps = ptdata.topinfo['fps'].iloc[i_top]
        top_arr = data_dict[data_dict_keys[i_top]]
        new_i_top = True
        if xticks_percent__sections:
            idx_isochrsec = ptdata.topinfo[f'trimmed_sections_{appaxis_lbl}s'].\
                            loc[data_dict_keys[i_top]]
        x_ticklabelling_dictargs['fps'] = fps
        x_ticklabelling_dictargs['idx_isochrsec'] = idx_isochrsec
        array_iterator = ndarr.diter( top_arr, lockdim=groupby )
        for vis_arr,i_ch,i_nd in array_iterator:
# TO-DO: Arg. 'topidxkey' (boolean) to add index and key of the top array to the title.
            sp_title = ''
            if new_i_top:
                if 'Name' in ptdata.topinfo.columns:
                    sp_title = '"'+ptdata.topinfo['Name'].iloc[i_top]+'"'
                elif 'Group' in ptdata.topinfo:
                    sp_title = 'Group: "'+ptdata.topinfo['Group'].iloc[i_top]+'"'
            plt.subplot(n_sp,1,i_sp)
            if sing_dims: vis_arr = np.squeeze(vis_arr)
            hax_len = vis_arr.shape[axes[-1]]-1
            x_ticklabelling_dictargs['hax_len'] = hax_len
            if vistype in ('line','cline','imshow'):
                if vistype in ('line','cline'):
                    dictargs_plot = {}
                    if vistype == 'cline':
                        dictargs_plot['marker'] = 'o'
                        dictargs_plot['markersize'] = dlattr_[2]*5
                    plt.plot( vis_arr.T, color=dlattr_[0], linestyle=dlattr_[1],
                              linewidth=dlattr_[2], **dictargs_plot )
                elif vistype == 'imshow':
                    if vis_arr.ndim != 2:
                        check_gb = abs(2-vis_arr.ndim)
                        exmsg = f'Number of dimensions for imshow is {vis_arr.ndim} but should \
                                be 2. Check that argument "groupby" has length = {check_gb}'
                        raise Exception(exmsg)
                    plt.imshow(vis_arr,aspect='auto',vmin=minmax[0],vmax=minmax[1])
                    plt.gca().invert_yaxis()
                plt.xlim((0,hax_len))
            elif 'spectrogram' in vistype:
                plt.specgram(vis_arr,Fs=fps,detrend='linear',scale='linear')
            if y_lim: plt.ylim(y_lim)
            if (y_ticks is not None) and (vis_arr.ndim == 2):
                if isinstance(y_ticks,list): sp_yticks.append(y_ticks)
                elif isinstance(y_ticks,dict): sp_yticks.append(y_ticks[ data_dict_keys[i_top] ])
                sp_axes.append( plt.gca() )
            plt.ylabel(ylabel)
            x_tick_labelling_( x_ticklabelling_dictargs )
            if sections and sections_appaxis_exist:
                vlsec = ptdata.topinfo[f'trimmed_sections_{appaxis_lbl}s'].iloc[i_top]
                if 'spectrogram' in vistype:
                    xstart,xend = plt.xlim()
                    vlsec = [ ( (v/hax_len)*(xend-xstart)+xstart ).item() for v in vlsec]
                    vlattr = vlattr.replace('k','w')
                overlay_vlines_( plt.gca(), vlsec, vlattr,
                                numcolour=[0.6,0.1,0.2], num_hvoffset=snum_hvoff )
            for i in i_ch:
                if isinstance(ptdata.labels.dimel[i],dict): # dict: different labels for each top array
                    sp_lbl = ptdata.labels.dimel[i][i_top][i_nd[i]]
                elif isinstance(ptdata.labels.dimel[i],list):
                    sp_lbl = ptdata.labels.dimel[i][i_nd[i]] # list: same labels for all top arrays
                else:
                    sp_lbl = ptdata.labels.dimel[i] # (singleton dim.) same labels for all top arrays
                sp_title = ''.join([sp_title,'\n',sp_lbl])
            if sp_title:
                plt.title(sp_title,y=spt_y)
            if new_i_top: new_i_top = False
            i_sp += 1
    fig.supxlabel(xlabel)
    plt.suptitle( super_title + figtitle , fontsize=16 )
    plt.tight_layout(rect=[0, 0.005, 1, 0.98])
# TO-DO: this might leave a bit too much space in between ticks:
    if (y_ticks is not None) and (vis_arr.ndim == 2):
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
