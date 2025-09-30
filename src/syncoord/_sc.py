'''
Functions that integrate multiple syncoord.ptdata functions of a same kind, and a data pipeline class.
They are imported by __init__.py, therefore they can be called directly with syncoord.
For example, to instantiate a PipeLine object: syncoord.PipeLine()
No need to do this: syncoord._sc.PipeLine()

'''

from copy import deepcopy

import numpy as np

from . import ptdata, ndarr, utils

# .............................................................................
# PUBLIC FUNCTIONS:

def halt( msg='halt' ):
    '''Useful for debugging.'''
    raise Exception(msg)

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
                cwt_freq (list,float): Frequency or [min, max] frequencies (Hz).
                postprocess (str): 'coinan' or None.
                If par['method'] == 'GXWT':
                    matlabeng (matlab,engine): object (useful when running multiple times).
                **kwargs (see documentation for "wct" and "gxwt" in module syncoord.ptdata)
            Optional:
                par['visint'] (dict,bool): Visualise intermediate steps. Default = False
                    True to use default parameters. Alternatively, a dict with keywords arguments
                    can be used and it will be passed to syncoord.ptdata.PtData.visualise()
    Returns:
        (syncoord.ptdata.PtData): Data out.
    '''
    def _check_CWT_par(ptd):
        if 'frequency' in ptdin.names.dim:
            raise Exception("Frequency-decomposed data is not suitable for CWT.")
    visint = par.pop('visint',False)

    if par['method'] == 'r':
        sync_1 = ptdata.kuramoto_r( ptdin )

    elif par['method'] == 'Rho':
        sync_1 = ptdata.rho( ptdin )

    elif par['method'] == 'PLV':
        plv_pairwise = ptdata.plv( ptdin, par['windows'] )
        _vis_dictargs(plv_pairwise, visint, None)
        sync_1 = ptdata.aggrax( plv_pairwise, function='mean' )

    elif par['method'] == 'WCT':
        _check_CWT_par(ptdin)
        if isinstance(par['cwt_freq'],list): minmaxf = par.pop('cwt_freq')
        else: minmaxf = [par['cwt_freq'], par.pop('cwt_freq')]
        if 'postprocess' not in par: par['postprocess'] = 'coinan'
        wct_pairwise = ptdata.wct( ptdin, minmaxf, 0, -1, **par )
        try:
            if visint: visint['savepath'] = visint['savepath'] + '_pairs'
        except: pass
        _vis_dictargs(wct_pairwise, visint, None)
        sync_1 = ptdata.aggrax( wct_pairwise, axis=0, function='mean' )

    elif par['method'] == 'GXWT':
        _check_CWT_par(ptdin)
        if isinstance(par['cwt_freq'],list): minmaxf = par.pop('cwt_freq')
        else: minmaxf = [par['cwt_freq']-0.01, par.pop('cwt_freq')+0.01]
        if ptdin.data[0].ndim == 3: fixed_axes = [-2,-1]
        elif ptdin.data[0].ndim == 2: fixed_axes = -1
        if 'postprocess' not in par: par['postprocess'] = 'coinan'
        gxwt_pairwise = ptdata.gxwt( ptdin, minmaxf, 0, fixed_axes, **par  )
        try:
            if visint: visint['savepath'] = visint['savepath'] + '_pairs'
        except: pass
        _vis_dictargs(gxwt_pairwise, visint, None)
        sync_1 = ptdata.aggrax( gxwt_pairwise, axis=0, function='mean' )

    if sync_1.names.dim[-2] == 'frequency':
        try:
            if visint: visint['savepath'] = visint['savepath'] + '_group'
        except: pass
        _vis_dictargs(sync_1, visint, None, vscale=1.3)
        sync_2 = ptdata.aggrax( sync_1, axis=-2, function='mean' )
    else: sync_2 = sync_1
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
                If func is 'secstats' the default for 'statnames' is 'mean".
    Returns:
        (syncoord.ptdata.PtData): Data out.
    '''
    funcs = ['secstats', 'corr']
    assert par['func'] in funcs, f"par['func'] = {par['func']} is invalid."
    kwargs = par.copy()
    [kwargs.pop(k) for k in ['func','vis']]

    if par['func'] == funcs[0]:
        if 'statnames' not in kwargs: kwargs['statnames'] = 'mean'
        yes_secmargins = ['$r$',r'$\rho$']
        not_secmargins = ['PLV','WCT','GXWT']
        if 'margins' not in kwargs:
            if ptdin.labels.main in ['$r$',r'$\rho$','PLV']: kwargs['margins'] = 'secsfromnan'
            elif ptdin.labels.main in ['WCT','GXWT']: kwargs['margins'] = None
            else: raise Exception(f'Invalild PtData.labels.main : {ptdin.labels.main}')
        ptdout = ptdata.secstats( ptdin, **kwargs )
    elif par['func'] == funcs[1]:
        par.pop('func')
        ptdout = ptdata.corr( ptdin, **kwargs )
    return ptdout

# .............................................................................
# PRIVATE FUNCTIONS:

def _vis_dictargs(ptd, vpar, key, **kwargs):
    if vpar:
        if key is None: vpar_n = vpar
        elif key in vpar: vpar_n = vpar[key]
        else: return
        if isinstance(vpar_n,dict): ptd.visualise(**vpar_n,**kwargs)
        elif vpar_n: ptd.visualise(**kwargs)
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
        other (dict): Multi-purpose container.
    Methods:
        run(): Runs the data pipeline.
    Args:
        (syncoord.ptdata.PtData): A PtData object with the input data.
        **kwargs:
            Arguments to syncoord.ptdata.load, to load data.
            AND/OR:
            matlab (matlab.engine,str,list): Either a matlab.engine object or path(s)
                for matlab functions.
    Returns:
        (syncoord.ptdata.PtData): Data out.
    '''
    def __init__(self,*args,**kwargs):
        if 'matlab' in kwargs: matlab = kwargs.pop('matlab')
        else: matlab = None
        steps = ["filt", "red1D", "phase", "sync","stats"]
        self.data = dict.fromkeys(steps)
        self.data['input'] = None
        if args:
            if isinstance(args[0], ptdata.PtData): self.data['input'] = args[0]
            elif kwargs: self.data['input'] = ptdata.load(*args,**kwargs)
        self.par = dict.fromkeys(steps)
        self.other = {}
        if matlab: self.other['matlab'] = matlab

    def run(self, stepar, gvis=None, sanitise=True, verbose=False):
        '''
        Run a pipeline accorptding to specified steps and their parameters.
        Args:
            stepar (dict): Ordered steps and their parameters for the pipeline. Steps are the
                dict's keys and parameters are a dict of keyword args or None. Steps are functions
                in syncoord.compo and are executed in the following order:
                    "filt", "red1D", "phase", "sync", "stats".
                The parameters' dict may include a dict "vis" with visualisation options,
                or True for default settings.
            Optional:
                gvis (dict) : Global visualisation options. Passed to syncooord.ptdata.visualise
                    gvis['pathfolder'] (bool): True = 'savepath' is a folder. Default = False
                sanitise (bool,str): Check for invalid steps in arg. "stepar". For example, if
                    stepar['sync']['method'] is 'WCT' or 'GXWT', "phase" is an invalid step.
                        If True: Invalid steps will be discarded from stepar.
                        If 'halt': An exception will be raised and the program will stop.
        Returns:
            (syncoord.ptdata.PtData): Resulting data.
        '''
        if sanitise:
            if stepar['sync']['method'] in ['GXWT']:
                if stepar['red1D'] not in [None,False]:
                    if sanitise is True: stepar['red1D'] = None
                    elif sanitise == 'halt': halt('"red1D" is an invalid step for GXWT')
            if stepar['sync']['method'] in ['WCT', 'GXWT']:
                if stepar['phase'] not in [None,False]:
                    if sanitise is True: stepar['phase'] = None
                    elif sanitise == 'halt': halt('"phase" is an invalid step for WCT and GXWT')

        if isinstance(gvis,dict):
            gvis_sw = True
            if ('pathfolder' in gvis) and ('savepath' in gvis):
                del gvis['pathfolder']
                folderpath = gvis['savepath']
            else: folderpath = None
        else: gvis_sw = False

        if stepar['sync']['method'] in ['GXWT']:
            if isinstance(self.other['matlab'],str) or isinstance(self.other['matlab'],list):
                self.other['matlab'] = utils.matlab_eng( addpaths=self.other['matlab'] )
            stepar['sync']['matlabeng'] = self.other['matlab']

        d = self.data['input']
        for st in self.par:
            assert st in stepar, f'"{st}" is not an allowed key'
            if gvis_sw and (stepar[st] is not None):
                if folderpath: gvis['savepath'] = f'{folderpath}/{st}'
                if ('vis' in stepar[st]) and (stepar[st]['vis'] is not None):
                    if stepar[st]['vis'] is True: stepar[st]['vis'] = gvis
                    else: stepar[st]['vis'] = {**stepar[st]['vis'],**gvis}
                if ('visint' in stepar[st]) and (stepar[st]['visint'] is not None):
                    if stepar[st]['visint'] is True: stepar[st]['visint'] = gvis
                    else: stepar[st]['visint'] = {**stepar[st]['visint'],**gvis}
            if stepar[st] == self.par[st]:
                if self.data[st]: d = self.data[st]
            else:
                if stepar[st] is not None:
                    if verbose: print('computing',st,'...')
                    fstr = st + '(d, stepar[st])'
                    d = eval(fstr)
                    self.par[st] = deepcopy(stepar[st])
                self.data[st] = d
            _vis_dictargs(d, stepar[st], 'vis')
        self.data['output'] = d
        return d
