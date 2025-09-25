'''Functions that integrate multiple ptdata functions of a same kind, and a data pipeline class.'''

from copy import deepcopy

import numpy as np
import scipy.stats as stats

from . import ptdata, ndarr, utils

# .............................................................................

def filt( ptdin, par ):
    '''Wrapper for syncoord.ptdata.smooth'''
    return ptdata.smooth(d, **par)

def red1D( ptdin, par ):
    '''
    Reduce to 1 dimension.
    Args:
        ptdin (syncoord.ptdata.PtData): Data in.
        par (dict):
            par['method'] (str): 'speed', 'norms', 'x', 'y', 'z'
            'dim' (int): Number of dimensions, the last in the array. Only for 'speed' or 'norms'
    '''
    if par['method'] == 'speed':
        return ptdata.apply( ptdin, sc.ndarr.tder, dim=par['dim' )
    elif par['method'] == 'norms':
        n1 = ptdata.norm( ptdin, order=1, axis=-par['dim'] )
        n2 = ptdata.norm( ptdin, order=2, axis=-par['dim'] )
        return ptdata.apply2( n1, n2, np.add )
    elif (par['method'] is not None) and (par['method'] in 'zyx'):
        return ptdata.select( ptdin, axis='zyx'.index(par['method']) )
    else: return ptdin

def phase( ptdin, par ):
    '''
    Phase angles.
    Args:
        ptdin (syncoord.ptdata.PtData): Data in.
        par (dict):
            par['method'] (str): 'FFT', 'peaks'
            If par['method'] == 'FFT':
                'fft_freq' (list, int): frequency or [min, max] frequencies (Hz).
            If par['method'] == 'peaks': Keyword arguments to syncoord.ptdata.peaks_to_phase
    '''
    if method == 'peaks':
        return ptdata.peaks_to_phase( ptdin, **kwargs )
    elif method == 'FFT':
        if isinstance(par['fft_freq'],list): fft_win_s = 1/par['fft_freq'][0]
        else: fft_win_s = 1/par['fft_freq']
        dout = ptdata.fourier( ptdin, fft_win_s, output='phase', mode='same' )
        if isinstance(par['fft_freq'],list):
            i_min_f = np.argmin(np.abs(par['fft_freq'][0] - phi.other['freq_bins'][0]))
            i_max_f = np.argmin(np.abs(par['fft_freq'][1] - phi.other['freq_bins'][0]))
            sel_freq_bin = slice(i_min_f, i_max_f+1)
        else: sel_freq_bin = 0
        return ptdata.select( phi, frequency=sel_freq_bin )

def gsync( ptdin, par ):
    '''
    Whole-group synchronisation.
    Args:
        ptdin (syncoord.ptdata.PtData): Data in.
        par (dict):
            par['method'] (str): 'r', 'Rho', 'PLV', 'WCT', 'GXWT'
            If par['method'] == 'PLV':
                windows (float,str): Window length in seconds for sliptding window or 'sections'.
            If par['method'] is in ['WCT','GXWT']:
                transfreq (list,float): Frequency or [min, max] frequencies (Hz).
                postprocess (str): 'coinan' or None.
                If par['method'] == 'GXWT':
                    matlabeng (matlab,engine): object (useful when running multiple times).
                **kwargs (see documentation for "wct" and "gxwt" in module syncoord.ptdata)
    '''

    mat_eng = par.get('mat_eng',None)

    if par['method']= 'r':
        sync_1 = ptdata.kuramoto_r( ptdin)
    elif par['method']= 'Rho':
        sync_1 = ptdata.rho( ptdin)
    elif par['method']= 'PLV':
        plv_pairwise = ptdata.plv( ptdin, par['windows'], mode='valid' )
        sync_1 = ptdata.aggrax( plv_pairwise, function='mean' )
    elif par['method']= 'WCT':
        if isinstance(par['transfreq'],list): minmaxf = par.pop('transfreq')
        else: minmaxf = [par['transfreq'], par.pop('transfreq')]
        wct_pairwise = ptdata.wct( ptdin, minmaxf, 0, -1, **par )
        sync_1 = ptdata.aggrax( wct_pairwise, axis=0, function='mean' )
    elif par['method']= 'GXWT':
        if isinstance(par['transfreq'],list): minmaxf = par.pop('transfreq')
        else: minmaxf = [par['transfreq']-0.01, par.pop('transfreq')+0.01]
        if ptdata.data[0].ndim == 3: fixed_axes = [-2,-1]
        elif ptdata.data[0].ndim == 2: fixed_axes = -1
        gxwt_pairwise = ptdata.gxwt( ptdin, minmaxf, 0, fixed_axes, **par  )
        sync_1 = ptdata.aggrax( gxwt_pairwise, axis=0, function='mean' )
    try:
        if sync_1.names.dim[-2] == 'frequency':
            sync_2 = ptdata.aggrax( sync_1, axis=-2, function='mean' )
        else: sync_2 = sync_1
    except: sync_2 = sync_1
    return sync_2

# def corrcont( ptdin, par ):
#     '''
#     Correlation between two arrays excluptding a margin at beginning and end, given by the number of
#     NaN in the first array.
#     Args:
#         ptdin (syncoord.ptdata.PtData): Data in. Should have only one top-level array, which is the first array
#             that may or may not have NaN toe stablish the margin.
#         par (dict):
#             par['arr'] (list,numpy.ndarr): The second array.
#             par['kind'] (str): Kind of correlation. Default = 'Kendall' (only currently available).
#     Returns:
#         coef (float): Correlation coefficient.
#         p (float): P-value.
#     '''
#     assert len(ptdin.data) == 1, 'ptdata should have only one top-level array'
#     k = list(ptdin.data.keys())[0]
#     for nan_margin,v in enumerate(ptdin.data[k]):
#         if not np.isnan(v): break
#     d_margin = nan_margin * 2
#     ptdata_d_valid = ptdata.data[0][slice(d_margin,len(ptdin.data[k])-d_margin-1)]
#     arr_d_valid = par['arr'][slice(d_margin,len(par['arr'])-d_margin-1)]
#     if len(ptdata_d_valid) < len(arr_d_valid): arr_d_valid = arr_d_valid[:-1]
#     coef, p = stats.kendalltau(ptdata_d_valid, arr_d_valid)
#     return coef, p

# def corrsecs( ptdata, par ):
#     '''
#     Correlation between the sections' means of an array, and a second array.
#     Args:
#         ptdin (syncoord.ptdata.PtData): Data in. Should have only one top-level array, and a "topinfo"
#             dataframe with sections.
#         par (dict):
#             par['arr'] (list,numpy.ndarr): The second array.
#             par['kind'] (str): Kind of correlation. Default = 'Kendall' (only currently available).
#     Returns:
#         coef (float): Correlation coefficient.
#         p (float): P-value.
#     '''
#     assert kind=='Kendall', "Only Kendall's rank correlation is available"
#     k = list(ptdin.data.keys())[0]
#     sync_secmeans = ptdata.secstats( ptdin, statnames='mean', last=True, omitnan=True )
#     coef, p = stats.kendalltau(sync_secmeans, par['arr'], nan_policy='omit')
#     return coef, p

def corr( ptdin, par ):
    '''
    Correlation, excluding NaN, between:
        1) Two 1-D arrays excluptding a margin at beginning and end, given by the number of
            NaN in the first array.
        2) The sections' means of a 1-D array, and a second 1-D array.
    Args:
        ptdata (syncoord.ptdata.PtData): The top-level arrays are the first 1-D array.
        par (dict):
            par['arr'] (list,numpy.ndarr): The second 1-D array.
            par['kind'] (str): Kind of correlation. Default = 'Kendall' (only currently available).
            par['sections'] (bool): If False, compute simple correlation. If True, compute
                                    correlation of sections' means
    Returns:
        coef (dict(key = float)): Correlation coefficients.
        p (dict(key = float)): P-values.
    '''
    assert par['kind']=='Kendall', "Only Kendall's rank correlation is available"
    if par['sections'] is True:
        ptdin = ptdata.secstats( ptdin, statnames='mean', last=True, omitnan=True )
    dout = {}
    for k in ptdin:
        coef[k], p[k] = stats.kendalltau(ptdin[k], par['arr'], nan_policy='omit')
    return coef, p

# .............................................................................

class PipeLine:
    '''
    Pipeline to quantify synchronisation.
    Attributes:
    Methods:
        run(**kwargs): Runs the data pipeline.
    Args:
        (syncoord.ptdata.PtData): A PtData object with the input data.
        OR
        Arguments to syncoord.ptdata.load, to load data.
    '''
    def __init__(self,*args,**kwargs):
        self.data = {}
        if isinstance(args[0], ptdata.PtData): self.data['input'] = args[0]
        else: self.data.['input'] = ptdata.load(*args,**kwargs)
        self.par = {}

    def run(self, stepar):
        '''
        Run a pipeline accorptding to specified steps and their parameters.
        Args:
            stepar (dict): Ordered steps and their parameters for the pipeline. Keys are steps and
                           parameters are a dict of keyword args or None.
                           Available steps are functions in syncoord.multi:
                               "filt", "red1D", "phase", "sync", "corrcont", "corrsecs".
        '''
        steps = ["filt", "red1D", "phase", "sync","corr"]
        d = self.data.input
        for k in stepar:
            assert k in steps, f'"{k}" is not an allowed key'
            for st in steps:
                if k == st:
                    try:
                        if stepar[k] != self.par[st]:
                            if stepar[k] is None: self.data[st] = None
                            else:
                                fstr = st + '(d, **stepar[k])'
                                self.data[st] = eval(fstr)
                            d = self.data[st]
                    except: d = self.data[st]
        self.data['output'] = d


