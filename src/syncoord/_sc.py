'''
Functions that integrate multiple syncoord.ptdata functions of a same kind, and a data pipeline class.
They are imported by __init__.py, therefore they can be called directly with syncoord.
For example, to instantiate a PipeLine object: syncoord.PipeLine()
No need to do this: syncoord._sc.PipeLine()

'''
import csv
import itertools
from os import path
from time import time
from datetime import timedelta
from copy import deepcopy

import numpy as np
import pandas as pd

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
        return ptdata.apply( ptdin, ndarr.tder, dim=par['dim'] )
    elif par['method'] == 'norms':
        n1 = ptdata.norm( ptdin, order=1, axis=-par['dim'] )
        n2 = ptdata.norm( ptdin, order=2, axis=-par['dim'] )
        return ptdata.apply2( n1, n2, np.add )
    elif (par['method'] is not None) and (par['method'] in 'zyx'):
        return ptdata.select( ptdin, axis='zyx'.index(par['method']) )
    else: return ptdin

def filtred( ptdin, par ):
    '''
    Filter and reduce to  1 dimension.
    '''
    return red1D( filt( ptdin, par ), par )

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

    if (len(sync_1.names.dim) > 1) and (sync_1.names.dim[-2] == 'frequency'):
        try:
            if visint: visint['savepath'] = visint['savepath'] + '_group'
        except: pass
        _vis_dictargs(sync_1, visint, None, vscale=1.3)
        sync_2 = ptdata.aggrax( sync_1, axis=-2, function='mean' )
    else: sync_2 = sync_1
    return sync_2

def stats( ptdin, par ):
    '''
    Apply one or more statistics functions from syncoord.ptdata
    Available functions are: 'secstats', 'corr'.
    The functions will be executed in the order as above.
    Args:
        ptdin (syncoord.ptdata.PtData): Data in.
        par (dict):
            par['func'] (str,list): Funcion or functions (see available functions above)
            Optional:
                kwargs: Arguments for the functions.
                    If func is 'secstats' the default for 'statnames' is 'mean".
                return_type (str): 'last' (default) to return only the last result of par['func']
                                    if it is a list; 'all' for a dict of results of all processes.
    Returns:
        If par['func'] is str or return_type is 'last':
            d (syncoord.ptdata.PtData): Data out.
        If par['func'] is list:
            stres (dict): Keys are as the input functions. Values are PtData objects.
    '''
    funcs = ['secstats', 'corr']
    if isinstance(par['func'],str): par['func'] = [par['func']]
    if 'return_type' in par: return_type = par.pop('return_type')
    else: return_type = 'last'
    for f in par['func']: assert f in funcs, f"par['func'] = {par['func']} is invalid."
    kwargs = par.copy()
    del kwargs['func']
    try: del kwargs['vis']
    except: pass
    d = ptdin
    stres = {}

    if 'secstats' in par['func']:
        if 'statnames' not in kwargs: kwargs['statnames'] = 'mean'
        yes_secmargins = ['$r$',r'$\rho$']
        not_secmargins = ['PLV','WCT','GXWT']
        if 'margins' not in kwargs:
            if d.labels.main in ['$r$',r'$\rho$','PLV']: kwargs['margins'] = 'secsfromnan'
            elif d.labels.main in ['WCT','GXWT']: kwargs['margins'] = None
            else: raise Exception(f'Invalild PtData.labels.main : {d.labels.main}')
        d = ptdata.secstats( d, **kwargs )
        stres['secstats'] = d

    if 'corr' in par['func']:
        arr = kwargs.pop('arr')
        if kwargs.get('cont',False):
            assert len(d.data) == 1, '"corr" "cont" only for PtData object with only one top-level array.'
            secs = d.topinfo.trimmed_sections_frames[0]
            shape = d.data[0].shape
            arr = ndarr.constant_secs(arr, secs, shape, last=True)
        d = ptdata.corr( d, arr, **kwargs )
        stres['corr'] = d

    if (len(par['func']) == 1) or (return_type == 'last'): return d
    elif return_type == 'all': return stres

# .............................................................................
# PRIVATE FUNCTIONS:

def _vis_dictargs(ptd, vpar, key, **kwargs):
    if vpar:
        if key is None: vpar_n = vpar
        elif key in vpar: vpar_n = vpar[key]
        else: return
        if isinstance(vpar_n,dict): spax = ptd.visualise(**vpar_n,**kwargs)
        elif vpar_n: spax = ptd.visualise(**kwargs)
        try: return spax
        except: pass

def _pvdisp(pval, sigdec=3):
    thresh = 10**-sigdec
    if pval < thresh: return f'p value < {thresh}'
    else: return f'p value = {round(pval,sigdec)}'
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
        self.data = dict.fromkeys(['input'] + steps + ['output'])
        self.__status = {'steps_init':False}
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
            stepar (dict): Parameters for steps of the pipeline. Steps are the
                dict's keys and parameters are a dict of keyword args or None, to functions
                in syncoord._sc which are executed in the following order:
                    "filt", "red1D", "phase", "sync", "stats"
                Steps "filt" and "red1D" may be combined in step "filtred".
                For details of the fucntions see help(syncoord.FUNCTION_NAME)
                The parameters' dict may include a dict "vis" with visualisation options,
                or True for default settings.
                Other optional arguments:
                    stepar['stats']['vis']['merge'] (bool): If True, the continuous group
                        synchronisation plot and the statistics plot will be merged.
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
        if self.__status['steps_init'] != True:
            if 'filtred' in stepar:
                steps = ["filtred","phase","sync","stats"]
                new_data_dict = dict.fromkeys(['input'] + steps + ['output'])
                new_data_dict['input'] = self.data['input']
                self.data = new_data_dict
                self.par = dict.fromkeys(steps)
        else: self.__status['steps_init'] = True

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

        try: vismrg = stepar['stats']['vis'].pop('merge')
        except: vismrg = False

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

            if ((st == 'sync') or (st == 'stats')) and (vismrg is True):
                stepar['stats']['return_type'] = 'all'
                if st == 'sync':
                    funcn = []
                    p = stepar['stats']['func']
                    for n in ['secstats','corr']:
                        funcn.append( (n in p) or (n in [p]))
                if st == 'stats':
                    try: printd = stepar[st]['vis']['printd']
                    except: printd = False
                    if funcn[1]:
                        corrkind = stepar[st].get('kind','Kendall')
                        if corrkind == 'Kendall': corrsymbol = 'Tau'
                        else: raise Exception(f'"kind" = {corrkind} not available')
                if stepar['stats'].get('cont',False):
                    assert sum(funcn) == 1, ''.join([ '"cont" for either "func" "secstats" or ',
                                                      '"corr", not both' ])
                    if st == 'sync':
                        if 'vis' not in stepar[st]: stepar[st]['vis'] = {}
                        stepar[st]['vis']['retspax'] = True
                        if funcn[0]: d_vis = d
                        elif funcn[1]:
                            d_vis = d.copy()
                            for k in d_vis.data:
                                d_vis.data[k] = ndarr.rescale(d_vis.data[k])
                        spax = _vis_dictargs(d_vis, stepar[st], 'vis')
                    elif st == 'stats':
                        for k in d.data:
                            assert len(spax[k]) == 1, ''.join([ 'Cannot merge visualisation of sync ',
                                'and stats when there are more than one dimensions of sync data.'])
                            ax = spax[k][0]
                            if funcn[0]:
                                if printd:
                                    n_frames = d.data[k].shape[-1]
                                    discrete_secs = []
                                    fr = d.topinfo['trimmed_sections_frames'][k]
                                    idx_secends = fr + [n_frames]
                                    i_start = 0
                                    for i_end in idx_secends:
                                        section = d.data[k][...,i_start:i_end]
                                        item = section[np.isfinite(section)][0].item()
                                        discrete_secs.append(item)
                                        i_start = i_end
                                    print(np.round(discrete_secs,3))
                                ax.plot(d.data[k],linewidth=3)
                            elif funcn[1]:
                                if printd:
                                    print(f'{corrsymbol} =',round(d.data[k][0],3))
                                    print(_pvdisp(d.data[k][1]))
                                arr = ndarr.rescale( stepar[st].pop('arr') )
                                secs = d.topinfo.trimmed_sections_frames[0]
                                shape = [int(ax.get_xlim()[1])]
                                carr = ndarr.constant_secs(arr, secs, shape, last=True)
                                nanmask = np.isnan(self.data['sync'].data[0])
                                carr[nanmask] = np.nan
                                ax.plot(carr,linewidth=2)
                else: # merged visualisation with discrete sections
                    if st == 'stats':
                        if sum(funcn) == 2:
                            dsc = ptdata.PtData(d['secstats'].topinfo)
                            for k in d['secstats'].data:
                                sync_secmeans_rs = ndarr.rescale( d['secstats'].data[k] )
                                arr_rs = ndarr.rescale(stepar[st]['arr'])
                                if printd:
                                    print(f'{corrsymbol} =',round(d['corr'].data[k][0],3))
                                    print(_pvdisp(d['corr'].data[k][1]))
                                dsc.data[k] = np.array([sync_secmeans_rs, arr_rs])
                            dsc_vis = { **gvis, 'vistype':'cline', 'sections':False,
                                        'x_ticklabelling':'index' }
                            try: del dsc_vis['savepath']
                            except: pass
                            spax = dsc.visualise( retspax=True, **dsc_vis )
                            lbl_1 = 'rescaled ' + stepar['sync']['method']
                            lbl_2 = 'rescaled ' + stepar['stats'].get('arrlbl','arr')
                            lbl_fs = gvis.get('fontsize',1) * 10
                            for k in d['secstats'].data:
                                ax = spax[k][0]
                                ax.legend([lbl_1, lbl_2], fontsize=lbl_fs)
                try: ax.figure.savefig(stepar[st]['vis']['savepath'] + '.png')
                except: pass
            else:
                if isinstance(d,dict):
                    for v in d.values(): _vis_dictargs(v, stepar[st], 'vis')
                else: _vis_dictargs(d, stepar[st], 'vis')
        self.data['output'] = d
        return d

def multicombo(*args,**kwargs):
    '''
    Run a data pipeline with multiple combinations of parameters.
    Args:
        (*args,**kwargs): Data arguments to PipeLine. See help(syncoord.PipeLine)
        itpar (dict): Iteration parameters and values for steps of the data pipeline, as a dict
                      with tuple keys:
                            itpar[('STEP','PAR','SUBPAR',...)] = values
                      For details of steps and their parameters see help(syncoord.PipeLine.run)
                      For each STEP there should be a PAR 'main' for the name of the main parameter,
                      some of which have corresponding PAR 'spec' (specifications) that have a
                      SUBPAR with the name of the main parameter. Other PAR are for parameters that
                      are common to all values of the main parameter.
                      Example:
                            itpar[('filtred','main')] = 'method'
                            itpar[('filtred','method')] = ['norms','speed','x']
                            itpar[('filtred','spec','norms','dim')] = ['2']
                            itpar[('filtred','spec','speed','dim')] = ['2']
                            itpar[('filtred','spec','speed','filter_type')] = ['savgol']
                            itpar[('filtred','spec','speed','window_size')] = 1
                            itpar[('filtred','spec','speed','order')] = [1, 2]
        rlbl (dict): Labels (str) for parameters and results to record in columns of a table. The
                     labels become headers of the table, in the same order as in the dict.
                     The results are from the last step of the pipeline. If more than one result is
                     produced for a combination of parameters, their corresponding labels should be
                     in a list. The keys are the same as for itpar, but with a prepended element:
                     'par' for parameter, or 'res' for result.
        results_folder (str): Folder where to save the resulting table.
        Optional:
            max_newres (int): Maximum number of new results. Useful for testing.
            verbose (int): 0 = Don't print anything (default);
                           1 = Print computation time and total number of results;
                           2 = As 1 and also print last result while running. Useful for testing.
    Returns:
        (pandas.DataFrame): Tabulated results.
    '''
    class breakloops(Exception): pass

    def _flat_unique_dict_values(d):
        def _notin(a,b):
            if a not in b: b.append(a)
            return b
        out = []
        for v in d.values():
            if isinstance(v,list):
                for e in v: out = _notin(e,out)
            else: out = _notin(v,out)
        return out

    def _make_sgrid(itpar, sm, STEPSW, final_step_lbl, sgrid_order):
        sgrid = {}
        for k_itpar in itpar:

            if sgrid_order < 3: final_step_sw = k_itpar[0] != final_step_lbl
            else: final_step_sw =  k_itpar[0] == final_step_lbl

            if (STEPSW[sm][k_itpar[0]] is True) and final_step_sw:

                if sgrid_order in [1,3]:
                    if 'spec' in k_itpar: get_this = False
                    else: get_this = True

                elif sgrid_order in [2,4]:
                    if 'main' in k_itpar:
                        mpar_val = itpar[(k_itpar[0],itpar[(k_itpar)])]
                        if sgrid_order == 2: get_this = True
                        else: get_this = False
                    elif ('spec' in k_itpar) and (mpar_val in k_itpar):
                        get_this = True
                    elif 'spec' not in k_itpar: get_this = True
                    else: get_this = False

                else: raise Exception('wrong sgrid_order')

                if get_this:
                    this_itpar = deepcopy(itpar[k_itpar])
                    if not isinstance(itpar[k_itpar],list): this_itpar = [this_itpar]
                    sgrid[k_itpar] = this_itpar
        return sgrid

    def _siter(itpar, STEPSW):

        for sm in itpar[('sync','method')]:
            final_step_lbl = tuple(STEPSW[sm].keys())[-1]
            sgrid_1 = _make_sgrid(itpar, sm, STEPSW, final_step_lbl, sgrid_order=1)

            # iterate mains, except final step:
            for v_1 in itertools.product(*sgrid_1.values()):
                iter_param_1 = {**itpar, **dict(zip(sgrid_1.keys(), v_1))}
                sgrid_2 = _make_sgrid(iter_param_1, sm, STEPSW, final_step_lbl, sgrid_order=2)

                # iterate specs, except final step:
                for v_2 in itertools.product(*sgrid_2.values()):
                    iter_param_2 = dict(zip(sgrid_2.keys(), v_2))
                    # iter_param_2: all parameters selected except for final step
                    sgrid_3 = _make_sgrid(itpar, sm, STEPSW, final_step_lbl, sgrid_order=3)

                    # iterate final step's mains:
                    for v_3 in itertools.product(*sgrid_3.values()):
                        iter_param_3 = {**itpar, **dict(zip(sgrid_3.keys(), v_3))}
                        sgrid_4 = _make_sgrid( iter_param_3, sm, STEPSW, final_step_lbl,
                                               sgrid_order=4 )

                        # iterate final step's specs:
                        for v_4 in itertools.product(*sgrid_4.values()):
                            iter_param_4 = {**iter_param_2, **dict(zip(sgrid_4.keys(), v_4))}
                            yield iter_param_4, final_step_lbl

    def _make_stepar(iter_param, STEPSW):
        sm = iter_param[('sync','method')]
        stepar = {st:{} for st,sw in STEPSW[sm].items() if sw is True}
        for k_sp, v_sp in iter_param.items():
            if k_sp[1] == 'spec':
                k_sp = tuple( v for i,v in enumerate(k_sp) if i not in [1,2] )
            if (STEPSW[sm][k_sp[0]] is True) and ('main' not in k_sp) and (len(k_sp) > 1):
                stepar[k_sp[0]][k_sp[1]] = v_sp
        return stepar

    def _append_results( result, all_results, iter_param, final_step_lbl, i_comb, gvars ):
        for arv in all_results.values():
            arv = arv.append('-')
        for k_sp, v_sp in iter_param.items():
            k_sp_list = list(k_sp)
            if k_sp[0] == final_step_lbl:
                k_res = tuple(['res'] + k_sp_list)
                if k_res in gvars['rlbl']:
                    these_lbl = gvars['rlbl'][k_res]
                    if isinstance(these_lbl,list):
                        for ie, e in enumerate(these_lbl):
                            all_results[ e ][-1] = result[ie]
                    else: all_results[ these_lbl ][-1] = result
            else:
                k_par = tuple(['par'] + k_sp_list)
                if k_par in gvars['rlbl']:
                    all_results[ gvars['rlbl'][k_par] ][-1] = v_sp
        all_results['i'][-1] = i_comb
        iter_result = [ all_results[h][-1] for h in gvars['headers'] ]
        if gvars['verbose'] == 2:
            iter_result_fmt = [ v if isinstance(v,str)
                                else '-' if v is None
                                else ':'.join([str(u) for u in v]) if isinstance(v,list)
                                else f'{v:,.3g}' for v in iter_result[1:] ]
            print(', '.join([str(i_comb)] + iter_result_fmt))
        return all_results, iter_result

    STEPSW = {}
    STEPSW['r'] = {'filtred':True,'phase':True,'sync':True,'stats':True}
    STEPSW['Rho'] = {'filtred':True,'phase':True,'sync':True,'stats':True}
    STEPSW['PLV'] = {'filtred':True,'phase':True,'sync':True,'stats':True}
    STEPSW['WCT'] = {'filtred':True,'phase':False,'sync':True,'stats':True}
    STEPSW['GXWT'] = {'filtred':False,'phase':False,'sync':True,'stats':True}

    gvars = {}
    itpar = kwargs.pop('itpar')
    strvar = kwargs.get('strvar',None)
    gvars['rlbl'] = kwargs.pop('rlbl')
    results_folder = kwargs.pop('results_folder')
    max_newres = kwargs.get('max_newres',None)
    gvars['verbose'] = kwargs.get('verbose',False)

    if not isinstance(itpar[('sync','method')],list):
        itpar[('sync','method')] = [itpar[('sync','method')]]
    if max_newres is None: max_newres = -1
    elif max_newres > 0: max_newres = max_newres-1
    else: raise Exception('max_newres should be None or greater than 0')

    gvars['headers'] = ['i'] + _flat_unique_dict_values(gvars['rlbl'])
    all_results = {k:[] for k in gvars['headers']}
    all_results_ffn = results_folder + '/all_results.csv'
    if gvars['verbose']: start_time = time()
    if path.isfile(all_results_ffn):
        with open(all_results_ffn,'r',newline='') as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # skip headers
            for i_row,v_row in enumerate(csv_reader):
                for k, v_cell in zip(gvars['headers'],v_row):
                    try:
                        if k == 'i': v_cell = int(v_cell)
                        else: v_cell = float(v_cell)
                    except: pass
                    all_results[k].append(v_cell)
            try:
                i_start = i_row
                compute = False
            except:
                i_start = 0
                compute = True
            new_file = False
    else:
        new_file = True
        i_start = 0
        compute = True

    pline = PipeLine(*args,**kwargs)

    i_comb = 0
    i_newres = 0
    try:
        with open(all_results_ffn,'a',newline='') as f:
            writer = csv.writer(f)
            if new_file: writer.writerow([s for s in gvars['headers']])
            all_comb = _siter(itpar, STEPSW)

            for ip, fsl in all_comb: # iterate through all combinations

                if compute:
                    stepar = _make_stepar(ip, STEPSW)
                    result_raw = pline.run(stepar)
                    result = result_raw.data[next(iter(result_raw.data))]
                    all_results, iter_result = _append_results( result, all_results, ip, fsl,
                                                                i_comb, gvars )
                    writer.writerow(iter_result)

                    if i_newres == max_newres: raise breakloops()
                    i_newres +=1
                else:
                    compute = i_comb == i_start
                i_comb += 1
    except breakloops: print(f'\nStopped at {i_newres+1} new results.')

    all_results_df = pd.DataFrame(all_results).set_index('i')
    if gvars['verbose']:
        toc = time() - start_time
        toc_fmt = timedelta(seconds=toc)
        print('computation time =',str(toc_fmt)[:-3])
        print(f'Total number of results = {all_results_df.shape[0]}')

    return all_results_df