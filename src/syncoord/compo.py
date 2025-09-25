'''Functions that integrate multiple ptdata functions of a same kind, and a data pipeline class.'''

from copy import deepcopy

import numpy as np

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

def stats( ptdin, par ): pass # ---------------------------------------------- DO THIS

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
                               "filt", "red1D", "phase", "sync", "stats".
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


