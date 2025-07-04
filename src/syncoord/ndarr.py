'''Functions that take numpy.ndarray as main input.'''

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.fft import rfft

from . import utils

# .............................................................................
# ANALYSIS-ORIENTED OPERATIONS:

def tder( arr_nd_in, dim=None, order=1 ):
    '''
    Differentiation of 1 dimension (axis -1) or more dimensions (axes -2 and -1).
    If more than one dimension, the first order difference is Euclidean distance among
    consecutive points. The second order difference is simple difference.
    Args:
        arr_nd_in (numpy.ndarray): 1-D array, or N-D array with 1 <= arr_nd_in.shape[-2] <= 3
        dim (int): number of dimensions to apply the time.derivative
        Optional:
            order (int): 1 (default) or 2
    Returns:
        (numpy.ndarray): array
    '''
    assert dim, "missing 1 required keyword argument: 'dim'"
    assert 1 <= arr_nd_in.shape[-2] <= 3, 'length of axis -2 should be 1, 2 or 3'
    assert 1 <= order <= 2, 'argument "order" should be 1 or 2'
    arr_nd_out = np.diff(arr_nd_in,axis=-1)
    if dim > 1: arr_nd_out = np.linalg.norm(arr_nd_out,axis=-2)
    if order == 2: arr_nd_out = np.diff(arr_nd_out)
    return arr_nd_out

def peaks_to_phase( arr_nd, endstart=False, axis=-1 ):
    '''
    Generate ramps between signal peaks, with amplitude {-pi,pi}
    Args:
        arr_nd (numpy.ndarray): N-D array
        Options:
            endstart (bool): True will add a ramp from zero before the first peak, and a ramp ending
                             in zero after the last peak. False will leave zeros before the first
                             peak and after the last peak.
            axis (int): Axis along which to execute the operation.
                Note: axis is a dimension of the N-D array. The axis that changes the most is -1
    Returns:
        (numpy.ndarray): N-D array
    '''
    def pks2ph(sig,endstart):
        len_sig = len(sig)
        phi = np.zeros(len_sig)
        idx_pks = signal.find_peaks(sig)
        for i in range(len(idx_pks[0])-1):
            i_start = idx_pks[0][i]
            i_end = idx_pks[0][i+1]
            ramp_length = int(np.diff(idx_pks[0][i:i+2])[0])
            phi[i_start:i_end] = np.linspace( start = -np.pi, stop = np.pi, num = ramp_length )
        if endstart:
            if idx_pks[0][0] != 0:
                phi[0:idx_pks[0][0]] = np.linspace(0, np.pi, idx_pks[0][0])
            if idx_pks[0][-1] != (len_sig-1):
                phi[idx_pks[0][-1]:-1] = np.linspace(-np.pi, 0, len_sig-idx_pks[0][-1]-1)
        return phi
    return np.apply_along_axis(pks2ph,axis,arr_nd,endstart)

def fourier_transform( arr_nd, window_length, fps=None, output='spectrum', window_shape=None,
                       mode='same', first_fbin=1, axis=-1 ):
    '''
    Wrapper for scipy.fft.rfft
    Fast Fourier transform for a signal of real numbers.
    Args:
        arr_nd (numpy.ndarray): N-D array
        window_length (int): length of the FFT window, in seconds if the fps parameter is given.
        Options:
            fps (int): frames per second
            output (str): 'phase'(radians), 'amplitude', 'spectrum' (complex)
            window_shape (str): name of the window shape (eg.'hann'). See help(scipy.signal.windows)
                                or https://docs.scipy.org/doc/scipy/reference/signal.windows.html
            mode (str): 'same' (post-process zero-padded) or 'valid'
            first_fbin (int): Remove frequency bins under this index. Default = 1 (DC offset).
            axis (int): Default = -1 (last dimension of the N-D array).
                Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        (numpy.ndarray): N-D array, whose two last two dimensions are the result of this function.
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
            elif output == 'amplitude': fft_result.append(np.abs(this_spectrum))
            elif output == 'phase': fft_result.append( np.angle(this_spectrum) )
        fft_result = np.array(fft_result).T
        if zpad:
            dif = len(sig) - fft_result.shape[1]
            margin = np.floor(dif/2).astype(int)
            fft_result = np.pad( fft_result, ( (0,0),(margin, margin + int(dif%2) )) )
        return fft_result
    return np.apply_along_axis( fourier_, axis, arr_nd, fft_window, window_length, zpad,
                                first_fbin, output )

def kuramoto_r(arr_nd):
    '''
    Row-wise kuramoto order parameter r.
    Args:
        arr_nd (numpy.ndarray): N-D array of phase angles, where dim = -2 is rows for points
                                and dim = -1 is columns for observations.
        Singleton dimensions will be removed firstly.
    Returns:
        (numpy.ndarray): 1-D array of Kuramoto order parameter r.
    '''
    if 1 in arr_nd.shape: arr_nd_sqz = np.squeeze(arr_nd)
    else: arr_nd_sqz = arr_nd
    n_pts = arr_nd_sqz.shape[-2]
    r_list = []
    for phi in arr_nd_sqz.T:
        r_list.append( abs( sum([(np.e ** (1j * angle)) for angle in phi]) / n_pts ) )
    return np.array(r_list)

def phasediff( phi_1, phi_2 ):
    '''
    Args:
        phi_1, phi_2 (numpy.ndarray,list): phase angles
    Returns:
        (numpy.ndarray): phase difference
    '''
    return np.arctan2( np.cos(phi_1) * np.sin(phi_2) - np.sin(phi_1) * np.cos(phi_2),
                       np.cos(phi_1) * np.cos(phi_2) + np.sin(phi_1) * np.sin(phi_2) )

def plv( a1, a2, axis=0 ):
    '''
    Phase-Locking Value for two vectors of phase angles.
    Args:
        a1, a2 (numpy.ndarray,list): phase angles.
        Optional:
            axis (int): Dimension to which the operation will be applied.
            Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        (numpy.ndarray): Phase-locking values.
    '''
    diff_complex = np.exp(complex(0,1)*(a1-a2))
    plv_result = np.abs(np.sum(diff_complex,axis=axis))/diff_complex.shape[axis]
    return plv_result

def windowed_plv( arrs, window_length=None, window_step=1, mode='same', axis=-1 ):
    '''
    Sliding-window phase-locking values.
    Args:
        arrs (numpy.ndarray,list[numpy.ndarray]): 1-D array or a list with two 1-D arrays
                                                  with the same length.
        window_length (int): Length of the window vector (frames).
        window_setp (int): Window step (frames).
        Optional:
            mode (str): 'same' (post-process zero-padded) or 'valid'.
            axis (int): Dimension to apply the process.
                Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        Array whose dimensions depend on func.
    '''
    return slwin( arrs, plv, window_length, window_step, mode=mode, axis=axis )

def xwt_nd( arrlist, minmaxf, fps, **kwargs ):
    '''
    Wrapper for Matlab functions cwt.m, cwtensor.m, genxwt.m, xwtnd.m
    Multi-Dimensional Cross-Wavelet Transform.
    Args:
        arrlist (list): Two N-D or 1-D arrays having dimensions [channels,frames]
                        or [frames], respectively.
                 Note: 'channels' are individual signals whose covariance will be measured.
                       For example, in the context of motion, each channel is a spatial
                       dimension (e.g., x, y).
        minmaxf (list[float]): Minimum and maximum frequency (Hz).
        fps (int): frames per second
        Optional:
            get_result (str): 'abs', 'angle', 'complex', 'real', 'imag'. Default = 'abs'
            projout (bool): Include power projections in returns.
            matlabeng (matlab.engine): object (useful when running multiple times).
                Otherwise the following arguments are valid:
            gxwt_path (str): path to folder containing functions cwtensor.m and genxwt.m
    Returns:
        result (numpy.ndarray): Array with dimensions [channels,frames], depending on argument 'get'.
        freqs (numpy.ndarray): frequencies (Hz).
        powproj (tuple(numpy.ndarray)): If projout = True, one array per with power projections.
                                        The arrays have dimensions [channels,frequencies,frames]
    Non-Python dependencies:
        Matlab, Wavelet Toolbox for Matlab, cwtensor.m, and genxwt.m
    Reference:
        https://doi.org/10.1016/j.humov.2021.102894
    '''
    ass_msg = 'First and second arguments should be lists with two elements.'
    assert isinstance(arrlist,list) and isinstance(minmaxf,list), ass_msg
    assert (len(arrlist)==2) and (len(minmaxf)==2), ass_msg

    get_result = kwargs.get('get_result','abs')
    projout = kwargs.get('projout',False)
    matlabeng = kwargs.get('matlabeng',None)
    gxwt_path = kwargs.get('gxwt_path',None)
    xwtnd_path = kwargs.get('xwtnd_path',None)
    verbose = kwargs.get('verbose',True)

    if projout: nout = 4
    else: nout = 2

    if matlabeng: neweng = False
    else:
        neweng = True
        addpaths = [gxwt_path,xwtnd_path]
        matlabeng = utils.matlab_eng(addpaths,verbose)

    xwtnd_result = matlabeng.xwtnd( arrlist[0].T, arrlist[1].T, float(fps),
                                    minmaxf[0], minmaxf[1], nargout=nout )
    result = np.array(xwtnd_result[0])
    if get_result == 'abs': result = np.abs(result)
    elif get_result == 'angle': result = np.angle(result)
    elif get_result == 'complex': pass
    elif get_result == 'real': result = np.real(result)
    elif get_result == 'imag': result = np.imag(result)
    else: raise Exception('value for argument "get_result" is invalid')
    result = np.flip(result, axis=0 )
    freqs = np.flip( np.squeeze( np.array(xwtnd_result[1]) ))
    output = [result,freqs]
    if projout:
        powproj = []
        for i in range(2,4):
            np_arr = np.array(xwtnd_result[i])
            if np_arr.ndim == 3:
                np_arr = np.moveaxis( np_arr, [0,1,2], [1,2,0] )
            powproj.append( np.flip( np.abs(np_arr)**2,axis=0 ) )
        output.append(tuple(powproj))

    if neweng:
        matlabeng.quit()
        if verbose: print('Disconnected from Matlab.')

    return tuple(output)

def isochronal_sections( data_list, idx_sections, last=False, axis=-1 ):
    '''
    Time-rescale 1-D data so that data fits into sections of the same size.
    The length of the resulting sections will be the length of the largest process axis of all
    N-D arrays.
    Args:
        data_list (list[numpy.ndarray]): N-D arrays with the data.
        idx_sections (list[list]): Index of sections for each N-D array.
        Optional kwargs:
            last (bool): If True, from the last index of sections to the end will be the last section.
            axis (int): Dimension to apply the process.
              Note: axis is a dimension of the N-D array. The rightmost axis (-1) is the fastest changing.
    Returns:
        isochr_data (list[numpy.ndarray]): processed data
        idx_isochr_sections (list): index of isochronal sections
    '''
    n_sections = np.inf
    for i in range(len(idx_sections)):
        if last:
            idx_sections[i] = idx_sections[i]+[len(data_list[i])]
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

def section_stats( arr_nd, idx_sections, fps, last=False, margins=None, axis=-1,
                   statnames=[ 'mean','median','min','max','std' ] ):
    '''
    Descriptive statistics for sections of an N-D array.
    Args:
        arr_nd (numpy.ndarray): N-D array
        idx_sections (list): index of the sections
        fps (int): frames per second
        Optional:
            last (bool): If True, last section starts at the last index.
            margins (float,list[float]). Trim at the beginning and ending, in seconds.
                     If float: Same trim bor beginning and ending.
                     If list: Trims for beginning and ending. Nested lists for sections.
            axis (int): Dimension to apply the process.
            statnames (str,list[str]): Statistics to compute. Default is all.
    Return:
        (numpy.ndarray): N-D array of same dimensions of the input arr_nd, except the two last
                         dimensions are [statistic, section]. Order as in argument 'statnames'.
    '''

    if isinstance(statnames,str): statnames = [statnames]
    n_stats = len(statnames)
    if 0 not in idx_sections: idx_sections = [0] + idx_sections
    if last and (arr_nd.shape[axis] not in idx_sections):
        idx_sections = idx_sections + [arr_nd.shape[axis]]
    n_sections = len(idx_sections)-1
    if margins is None:
        margins_f = None
    elif isinstance(margins,list):
        if isinstance(margins[0],list):
            margins_f = [ [ round(v*fps) for v in nl ] for nl in margins ]
        else:
            mmf = [ round(margins[0] * fps), round(margins[1] * fps) ]
            margins_f = [ mmf for i in range(n_sections) ]
    else:
        mf = round(margins * fps)
        margins_f = [ [mf,mf] for i in range(n_sections) ]
    def sstats_( arr_1d_in, n_sections, statnames, n_stats, margins_f ):
        arr_1d_out = np.empty((n_stats,n_sections))
        for i_stat in range(n_stats):
            this_statname = statnames[i_stat]
            for i_sec in range(n_sections):
                i_sec_start = idx_sections[i_sec]
                i_sec_end = idx_sections[i_sec+1]
                this_section = arr_1d_in[i_sec_start:i_sec_end]
                if margins_f:
                    i_trim_start = margins_f[i_sec][0]
                    i_trim_end = len(this_section)-margins_f[i_sec][1]
                    this_section = this_section[i_trim_start:i_trim_end]
                if this_statname == 'mean':
                    arr_1d_out[i_stat,i_sec] = np.mean(this_section)
                elif this_statname == 'median':
                    arr_1d_out[i_stat,i_sec] = np.median(this_section)
                elif this_statname == 'min':
                    arr_1d_out[i_stat,i_sec] = np.min(this_section)
                elif this_statname == 'max':
                    arr_1d_out[i_stat,i_sec] = np.max(this_section)
                elif this_statname == 'std':
                    arr_1d_out[i_stat,i_sec] = np.std(this_section)
        return arr_1d_out

    return np.apply_along_axis( sstats_, axis, arr_nd, n_sections, statnames, n_stats, margins_f )

# .............................................................................
# APPLICATION:

def diter( arr_nd, lockdim=None ):
    '''
    Generator that iterates over dimensions of an N-D data array.
    Args:
        arr_nd (numpy.ndarray,tuple): N-D data array (row major), or tuple with shape of an array.
        Optional kwarg:
            lockdim (int,list[int]): Dimension(s) to not iterate over.
    Returns:
        arr_out (numpy.ndarray): slice of d corresponding to the iteration, only if d is np.ndarray.
                                 If arr_nd is a tuple this is not returned.
        i_chdim (list): Index of changing dimensions.
        multi_idx (list): Multi-index of the current slice. If lockdim is not None,
                          elements at lockdim are replaced with ':'.
    '''
    if isinstance(arr_nd,tuple):
        s = list(arr_nd)
        disarray = False
    elif isinstance(arr_nd,np.ndarray):
        s = list(arr_nd.shape)
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
            arr_out = eval('arr_nd'+multi_idx_str)
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

def slwin( arrs, func, window_length, window_step=1, mode='same', **kwargs ):
    '''
    Apply a function to a sliding window over the last dimension (-1) of one or two numpy arrays.
    Args:
        arrs (numpy.ndarray,list[numpy.ndarray]): One or two 1-D or N-D array(s) with same dimensions.
        func (Callable): Function to apply, with one or two required inputs (consistent with arrs).
        window_length (int): Length of the window vector.
        window_step (int): Step or "hop" of the moving window.
        Optional kwargs:
            mode (str): 'same' (post-process zero-padded) or 'valid'.
            **kwargs = keyword arguments to be passed to func.
    Returns:
        Array whose dimensions depend on func.
    '''

    if isinstance(arrs,list): len_arr = arrs[0].shape[-1]
    else: len_arr = arrs.shape[-1]
    slwin_result = []
    m_bit = 0
    for i_window in range(0,len_arr-window_length,window_step):
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
        dif = len_arr/window_step - slwin_result.shape[-1]
        margin = np.floor(dif/2).astype(int)
        pad_width = [(margin, margin + int(dif%2))]
        if slwin_result.ndim > 1:
            for _ in range(slwin_result.ndim-1): pad_width.insert(0,(0,0))
        slwin_result = np.pad(slwin_result,tuple(pad_width))
    return slwin_result

def apply_to_pairs( arr_nd, func, pairs_axis, fixed_axes=-1, imout=0, verbose=False, **kwargs ):
    '''
    Apply a function to pairs of dimensions of an N-D array.
    Args:
        arr_nd (numpy.ndarray): N-D array.
        func (Callable): Function to apply. Its first argument is a list with N-D arrays of the pair.
        pairs_axis (int): Dimension to run the pairwise process.
        Optional kwargs:
            fixed_axes (int,list[int]): Dimension(s) that are the input to func. Should be equal or
                        less than the dimensions in the output array of func, and should not include
                        pairs_axis. Default is last axis *.
            imout (int): Index of N-D array in returned tuple of func, if func has multiple returns.
            verbose (bool): Display progress.
            **kwargs = optional arguments and keyword arguments passed to func.
    Returns:
        arr_nd_out (numpy.ndarray): N-D array. The length of the pairs dimension originanlly
                                    of length N, is ((N*N)-N)/2.
        pairs_idx (list): pairs
        multi_results (tuple): func returns other than indicated by argument 'imout', otherwise
                               empty. Consecutively equal results will be discarded.
        new_fixed_axes (list)
    * axes = dimensions of the N-D array, where the rightmost axis is the fastest changing.
    '''
    shape_in = list(arr_nd.shape)
    iter_shape = shape_in.copy()
    n_pair_el = iter_shape.pop(pairs_axis)
    n_pairs = (n_pair_el**2 - n_pair_el)//2
    loc_idx_iter = list(range(len(shape_in)))
    del loc_idx_iter[pairs_axis]
    if not isinstance(fixed_axes,list): fixed_axes = [fixed_axes]
    fixed_axes.sort()
    new_fixed_axes = fixed_axes.copy()
    idx_shape_o = [None for _ in shape_in]
    for i in fixed_axes:
        del iter_shape[i]
        del loc_idx_iter[i]
        idx_shape_o[i] = ':'
    idx_shape_i = idx_shape_o.copy()
    idx_shape_j = idx_shape_o.copy()
    shape_out = shape_in.copy()
    shape_out[pairs_axis] = (shape_in[pairs_axis]**2 - shape_in[pairs_axis])//2
    arr_nd_out = np.empty(tuple(shape_out))
    shape_fixed_axes = tuple(shape_out[v] for v in fixed_axes)
    len_pairs_axis = arr_nd.shape[pairs_axis]
    i_pair = 0
    pairs_idx = []
    multi_results = []
    def checkmr_(a,b):
        if isinstance(a, list): a = a[0]
        if isinstance(b, list): b = b[0]
        try: return (a.all() != b.all()) or (a.shape != b.shape)
        except: return True
    arr_nd_out_mutated = False
    for i in range(len_pairs_axis):
        idx_shape_i[pairs_axis] = i
        for j in range(i+1, len_pairs_axis):
            idx_shape_j[pairs_axis] = j
            idx_shape_o[pairs_axis] = i_pair
            if verbose: print(f'pair {i_pair+1} of {n_pairs}')
            pairs_idx.append([i,j])
            for idx_iter in np.ndindex(tuple(iter_shape)):
                for i_loc,i_idx in zip(loc_idx_iter,idx_iter):
                    idx_shape_i[i_loc] = i_idx
                    idx_shape_j[i_loc] = i_idx
                    idx_shape_o[i_loc] = i_idx
                idx_shape_i_str = str(idx_shape_i).replace("'","")
                idx_shape_j_str = str(idx_shape_j).replace("'","")
                idx_shape_o_str = str(idx_shape_o).replace("'","")
                slice_i = eval('arr_nd'+idx_shape_i_str)
                slice_j = eval('arr_nd'+idx_shape_j_str)
                func_result = func([slice_i,slice_j],**kwargs)
                if isinstance(func_result,np.ndarray):
                    result_arr = func_result
                elif isinstance(func_result,tuple):
                    result_arr = func_result[imout]
                    other_results = [a for i,a in enumerate(func_result) if i is not imout]
                    if i_pair != 0:
                        other_results = [ a for a,b in zip(other_results,multi_results[i_pair-1])
                                          if checkmr_(a,b) ]
                    multi_results.append(other_results)
                if result_arr.shape != shape_fixed_axes:
                    if arr_nd_out_mutated:
                        raise Exception('func not returning arrays with consistent shape.')
                    fixed_axes_pos = fixed_axes.copy()
                    for i_fap,a in enumerate(fixed_axes_pos):
                        if a < 0: fixed_axes_pos[i_fap] = arr_nd_out.ndim + a
                    shape_out_new = shape_out.copy()
                    if len(fixed_axes_pos)==1:
                        if result_arr.ndim==1:
                            shape_out_new[fixed_axes_pos[0]] = result_arr.shape
                        else:
                            shape_out_new[fixed_axes_pos[0]:fixed_axes_pos[0]+1] = result_arr.shape
                            idx_shape_o[fixed_axes_pos[0]:fixed_axes_pos[0]+1] = \
                                [':' for _ in range(result_arr.ndim)]
                            new_fixed_axes = fixed_axes_pos \
                                + [fixed_axes_pos[0]+i+1 for i in range(result_arr.ndim)]
                    elif len(fixed_axes_pos) == result_arr.ndim:
                        for i_n, i_fap in enumerate(fixed_axes_pos):
                            shape_out_new[i_fap] = result_arr.shape[i_n]
                            idx_shape_o[i_fap] = ':'
                    else: raise Exception('values in argument "fixed_axes" should be less')
                    idx_shape_o_str = str(idx_shape_o).replace("'","")
                    arr_nd_out = np.empty(tuple(shape_out_new))
                    shape_fixed_axes = result_arr.shape
                    arr_nd_out_mutated = True
                exec('arr_nd_out'+idx_shape_o_str+' = result_arr')
                i_pair += 1
    if multi_results: return arr_nd_out, pairs_idx, new_fixed_axes, multi_results
    else: return arr_nd_out, pairs_idx, new_fixed_axes

def apply_dimgroup( arr_in, func, exaxes=None, i_out=0, n_out='all' ):
    '''
    Group dimensions and apply a function: func( arr_2D,*args,**kwargs),
    where arr_2D is a 2-dimensional array, and func returns a 1-dimensional array.
    Args:
        arr_in (numpy.ndarray): input N-D array
        func (Callable): function to apply
        Optional:
            exaxes (int): Dimension(s) to exclude from grouping, except last dimension.
                          If None, all dimensions except last will be grouped.
            i_out (int,str): If int, index of function's return to cast to output N-D array.
                             If str, tuple to return the function's return only if arr_in is
                             2-dimensional or if exaxes=None.
            n_out (int): number of elements in last dimension of function's output: 'all' or 1.
    Returns:
        output (numpy.ndarray,tuple)
    '''
    if arr_in.ndim < 2:
        raise Exception('input dimensions should be at least 2')
    else:
        assert (n_out=='all') or (n_out==1), "".join([ "n_out can only be 'all' or 1 for ",
                                                       "input arrays of more than 2 dimensions" ])
        if arr_in.ndim > 2:

            def _reshape_exlast(arr_a):
                arrdim = 1
                for n in arr_a.shape[:-1]: arrdim *= n
                return np.reshape(arr_a,(arrdim,arr_a.shape[-1]))

            if exaxes is None:
                arr_1 =  _reshape_exlast(arr_in)
                funcret = func(arr_1)
                if isinstance(funcret,tuple) or (i_out != 'tuple'): output = funcret[i_out]
                else: output = funcret
            else:
                graxes, exaxes = utils.invexaxes(exaxes, arr_in.shape, arr_in.ndim)
                shape_out = []
                for i in exaxes: shape_out.append(arr_in.shape[i])
                if n_out == 1: shape_out.append(1)
                else: shape_out.append(arr_in.shape[-1])
                output = np.empty(shape_out)
                array_iterator = diter( arr_in, lockdim=graxes )
                for i,item in enumerate(array_iterator):
                    arr_0,i_ch,i_nd = item
                    arr_1 =  _reshape_exlast(arr_0)
                    idxex_out = [v for v in i_nd if v != ':']
                    funcret = func(arr_1)
                    output[idxex_out,:] = funcret[i_out]
        else:
            funcret = func(arr_in)
            if isinstance(funcret,tuple) or (i_out != 'tuple'): output = funcret[i_out]
            else: output = funcret
    return output