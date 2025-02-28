'''Functions that take numpy.ndarray as main input.'''

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.fft import rfft

def iter(ndarr,lockdim=None):
    '''
    Generator that iterates over dimensions of an N-ndarray array.
    Args:
        ndarr: N-D array (C-style, row major), or tuple with the shape of an array.
        Optional:
            lockdim: dimension (int) or dimensions (list) to not iterate over.
    Returns:
        arr_out: slice of d corresponding to the iteration, only if d is np.ndarray. 
                 If d is a tuple this is not returned.
        i_chdim: (list) index of changing dimensions.
        multi_idx: (list) multi-index of the current slice. If lockdim is not None, 
                   elements at lockdim are replaced with ':'.
    '''
    if isinstance(ndarr,tuple):
        s = list(ndarr)
        disarray = False
    elif isinstance(ndarr,np.ndarray):
        s = list(ndarr.shape)
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
            arr_out = eval('ndarr'+multi_idx_str)
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

def fourier_transform( ndarr, window_length, fps=None, output='spectrum', window_shape=None,
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
        from importlib import import_module
        windows_module = import_module('scipy.signal.windows')
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

def apply_to_pairs(ndarr, func, pairs_axis, fixed_axes=-1, verbose=True, **kwargs):
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

def windowed_plv(arrs, window_length=None, mode='same', axis=1):
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

