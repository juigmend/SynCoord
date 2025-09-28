'''
Functions that integrate multiple syncoord.ptdata functions of a same kind, and a data pipeline class. The functions and the class can be called directly with syncoord. (no need for .compo), because all objects of this module (compo) are imported by __init__.py
'''

from copy import deepcopy

import numpy as np

from . import ptdata, ndarr, utils

# .............................................................................

def filt( ptdin, par ):
    '''Wrapper for syncoord.ptdata.smooth'''
    return ptdata.smooth(ptdin, **par)

def red1D( ptdin, par ):
    '''
    Reduce to 1 dimension.
    Args:
        ptdin (syncoord.ptdata.PtData): Data in.
        par (dict):
            par['method'] (str): 'norms', 'speed', 'x', 'y', 'z'
            'dim' (int): Number of dimensions, the last in the array. Only for 'norms' or 'speed'.
    Returns:
        (syncoord.ptdata.PtData): Data out.
    '''
    if par['method'] == 'speed':
        return ptdata.apply( ptdin, sc.ndarr.tder, dim=par['dim'] )
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
    Returns:
        (syncoord.ptdata.PtData): Data out.
    '''
    if par['method'] == 'peaks':
        pdout = ptdata.peaks_to_phase( ptdin, **kwargs )
        margin = kwargs.get('min_dist',None)
    elif par['method'] == 'FFT':
        if isinstance(par['fft_freq'],list): fft_win_s = 1/par['fft_freq'][0]
        else: fft_win_s = 1/par['fft_freq']
        phi = ptdata.fourier( ptdin, fft_win_s, output='phase', mode='same' )
        if isinstance(par['fft_freq'],list):
            i_min_f = np.argmin(np.abs(par['fft_freq'][0] - phi.other['freq_bins'][0]))
            i_max_f = np.argmin(np.abs(par['fft_freq'][1] - phi.other['freq_bins'][0]))
            sel_freq_bin = slice(i_min_f, i_max_f+1)
        else: sel_freq_bin = 0
        pdout = ptdata.select( phi, frequency=sel_freq_bin )
        margin = fft_win_s
    pdout.other['phase'] = { 'margin' : margin }
    return pdout

def sync( ptdin, par ):
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
    Returns:
        (syncoord.ptdata.PtData): Data out.
    '''
    mat_eng = par.get('mat_eng',None)

    if par['method'] == 'r':
        sync_1 = ptdata.kuramoto_r( ptdin )
    elif par['method'] == 'Rho':
        sync_1 = ptdata.rho( ptdin )
    elif par['method'] == 'PLV':
        plv_pairwise = ptdata.plv( ptdin, par['windows'], mode='valid' )
        sync_1 = ptdata.aggrax( plv_pairwise, function='mean' )
    elif par['method'] == 'WCT':
        if isinstance(par['transfreq'],list): minmaxf = par.pop('transfreq')
        else: minmaxf = [par['transfreq'], par.pop('transfreq')]
        wct_pairwise = ptdata.wct( ptdin, minmaxf, 0, -1, **par )
        sync_1 = ptdata.aggrax( wct_pairwise, axis=0, function='mean' )
    elif par['method'] == 'GXWT':
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

def stats( ptdin, par ):
    '''
    Apply statistics functions from syncoord.ptdata
    Available functions are: 'secstats', 'corr'.
    Args:
        ptdin (syncoord.ptdata.PtData): Data in.
        par (dict):
            par['func'] (str): Funcion (see available functions above)
            Optional:
                kwargs: Arguments for the functions.
                If func is 'secstats' the default for "statnames" is "mean".
    Returns:
        (syncoord.ptdata.PtData): Data out.
    '''
    funcs = ['secstats', 'corr']
    assert par['func'] in funcs, f"par['func'] = {par['func']} is invalid."
    kwargs = par.copy()
    [kwargs.pop(k) for k in ['func','vis']]

    if par['func'] == funcs[0]:
        if 'statnames' not in kwargs: kwargs['statnames'] = 'mean'
        ptdout = ptdata.secstats( ptdin, **kwargs )
    elif par['func'] == funcs[1]:
        par.pop('func')
        ptdout = ptdata.corr( ptdin, **kwargs )
    return ptdout

# .............................................................................

class PipeLine:
    '''
    Pipeline to quantify synchronisation.
    Attributes:
        data (dict): Should have at least item "input" created at initialisation with a PtData
                     object as argument, or arguments for syncoord.ptdata.load
                     When the "run" method is executed, more an item for the ouput of each step
                     will be added, and item "output".
        par (dict): Contains parameters for each step. Has the same keys as attribute "data",
                    except "input" and "output".
    Methods:
        run(): Runs the data pipeline.
    Args:
        (syncoord.ptdata.PtData): A PtData object with the input data.
        OR
        Arguments to syncoord.ptdata.load, to load data.
    Returns:
        (syncoord.ptdata.PtData): Data out.
    '''
    def __init__(self,*args,**kwargs):
        self.data = {}
        if isinstance(args[0], ptdata.PtData): self.data['input'] = args[0]
        else: self.data['input'] = ptdata.load(*args,**kwargs)
        self.par = dict.fromkeys(["filt", "red1D", "phase", "sync","stats"])

    def run(self, stepar):
        '''
        Run a pipeline accorptding to specified steps and their parameters.
        Args:
            stepar (dict): Ordered steps and their parameters for the pipeline. Keys are steps and
                           parameters are a dict of keyword args or None. Steps are functions in
                           syncoord.compo and are executed in this order:
                               "filt", "red1D", "phase", "sync", "stats".
                           The parameters' dict may include a dict "vis" with visualisation options,
                           or True for default settings.
        Returns:
            (syncoord.ptdata.PtData): Data out.
        '''
        d = self.data['input']
        for st in self.par:
            assert st in stepar, f'"{st}" is not an allowed key'
            if stepar[st] != self.par[st]:
                if stepar[st] is not None:
                    fstr = st + '(d, stepar[st])'
                    d = eval(fstr)
                self.data[st] = d
            if (stepar[st] is not None) and ('vis' in stepar[st]) \
            and (stepar[st]['vis'] is not None):
                if isinstance(stepar[st]['vis'],dict): visarg = stepar[st]['vis']
                else: visarg = None
                if visarg: d.visualise(**visarg)
                else: d.visualise()
        self.data['output'] = d
        return d
