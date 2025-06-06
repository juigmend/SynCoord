'''Miscellaneous functionality.'''

import os
from copy import deepcopy

import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

def nanbordz(a,margin):
    '''Converts zeros to nan within a margin at beginning and ending of a vector'''
    for am in [a[:margin],a[-margin:]]: am[am == 0] = np.nan
    return a

def frames_to_minsec_str( frames, fps, ms=False ):
    '''
    Convert frames (int) into a string with format "minutes:seconds" (default)
    or "minutes:seconds.miliseconds".
    Args:
        frames (scalar): number of frames
        fps (int): frames per second
        Optional:
            ms (bool): include miliseconds
    Returns:
        Formatted string.
    '''
    if ms: rounder_ = int
    else: rounder_= round
    mins, sec = divmod((frames/fps), 60)
    sec_r = rounder_(sec)
    if sec == 0:
        lbl = f'{int(mins)}:00'
    elif sec < 10:
        lbl = f'{int(mins)}:0{sec_r}'
    else:
        lbl = f'{int(mins)}:{sec_r}'
    if ms:
        lbl = f'{lbl}.{int(round(sec-sec_r,3)*1000)}'
    return lbl

def minsec_str_to_frames(ts_str_in,fps):
    '''
    Args:
        ts_str_in: comma-separated string of times as seconds, minutes:seconds, or minutes:seconds.decimals
        fps
    Returns:
        list of int: frames
    '''
    if not ts_str_in[-1].isnumeric():
        ts_str_in = ts_str_in[:-1]
    ts_str_in = ts_str_in.replace(' ','')
    ts_str_lst = ts_str_in.split(',')
    ts_f_lst = []
    for ts_str in ts_str_lst:
        if ':' not in ts_str:
            s = float(ts_str)
        else:
            s = sum(x * float(t) for x, t in zip([60, 1], ts_str.split(":")))
        ts_f_lst.append( round(s*fps) )
    return ts_f_lst

def trim_sections_to_frames(topinfo):
    '''
    Args:
        DataFrame with columns 'fps', and 'Sections'. If 'Sections' does not exits, then
        values in column 'trimmed_sections_frames' will be used.
        Optionally the dataframe can have column 'Start' to indicate trim at the beginning.
        See documentation for syncoord.utils..minsec_str_to_frames, for the format of cells
        in column 'Sections'.
    Returns:
        List with processed values.
    '''
    ts_f = []
    for k in topinfo.index:
        if 'Sections' in topinfo:
            s_f = minsec_str_to_frames( topinfo.loc[k,'Sections'] , topinfo.loc[k,'fps'] )
        else:
            s_f = topinfo.loc[k,'trimmed_sections_frames']
        if 'Start' in topinfo: offset_s = topinfo.loc[k,'Start']
        else: offset_s = 0
        if np.isnan(offset_s): offset_s = 0
        offset_f = offset_s * topinfo.loc[k,'fps']
        ts_f.append( [ round(f - offset_f) for f in s_f ] )
    return ts_f

def trim_topinfo_start( ptdata, trim_s ):
    '''
    Modifies ptdata.topinfo such that the sections in frames reflect a negative offset in time.
    Args:
        ptdata object
        trim_s: trim at the beginning, in seconds.
    '''
    if 'Start' in ptdata.topinfo.columns:
        topinfo = deepcopy(ptdata.topinfo)
        old_start = topinfo['Start']
        topinfo['Start'] = trim_s
        try:
            topinfo['trimmed_sections_frames'] = trim_sections_to_frames(topinfo)
        except KeyError: pass
        topinfo['Start'] = old_start + trim_s
    else:
        topinfo = ptdata.topinfo.assign( Start =
                                         [trim_s for _ in range(ptdata.topinfo.shape[0])] )
        try: topinfo['trimmed_sections_frames'] = trim_sections_to_frames(topinfo)
        except KeyError: pass
    return topinfo

def supersine(argdict):
    '''
    Produce a sine wave with optional variables for distortion.
    Arg:
            wavargs = dict()
        Required values:
            wavargs['frequency'] = f
            wavargs['phase_shift'] = ps
            wavargs['amplitude'] = a
            wavargs['sampling_frequency'] = sf
            wavargs['length'] = l
        Optional values:
            wavargs['vertical_offset'] = vo
            wavargs['irregularity'] = irr
            wavargs['noise_strength'] = ns
            wavargs['seed'] = s
    Returns:
        array
    '''
    from scipy.interpolate import CubicSpline

    f = argdict.get('frequency')
    ps = argdict.get('phase_shift')
    amp = argdict.get('amplitude')
    fps = argdict.get('sampling_frequency')
    l = argdict.get('length')
    v_offset = argdict.get('vertical_offset',0)
    irregularity = argdict.get('irregularity',0)
    noise_strength = argdict.get('noise_strength',0)
    noise_strength = noise_strength
    seed = argdict.get('seed',None)

    rangen = np.random.default_rng(seed=seed)
    t = np.arange(0, l)/fps
    y = np.sin(2*np.pi*f*t + ps/2)
    if irregularity:
        if irregularity >= 1:
            y = np.zeros(l)
        else:
            n_ran = int(np.ceil((l-2) * (1-irregularity)**2))
            if n_ran < 1: n_ran = 1
            ran_idx = np.sort(rangen.choice(np.arange(1,l), size=n_ran, replace=False))
            ip_x = np.insert(ran_idx, [0,n_ran], [0,l])
            ip_y = y[ran_idx]
            ip_y = ip_y - ip_y * rangen.random(size=ip_y.shape[0]) * irregularity**2
            ip_y = np.insert(ip_y, [0,n_ran], [0,0])
            ipf = CubicSpline(ip_x,ip_y)
            y = ipf(np.arange(l))
    noiz = rangen.uniform(-0.9,0.9,l)
    y = (y - (y * noiz * noise_strength ))
    y = (y * amp) / (np.max( [ abs(y.min()) , y.max() ] ) * 2)
    y = y + v_offset
    return y

def init_testdatavars(**kwargs):
    '''
    Initialise values for syncoord.utils.testdata, which produces signals of oscillating points
    with optional distortion.
    All arguments are keywords. If no arguments are given, a dictionary with default values
    will be produced (e.g., for quick testing).
    Args:
        Optional:
            fps: scalar
            durations_sections: list, durations of sections in seconds
            n_points: number of oscillating points
            n_axes: number of spatial axes
            seed: None or int; seed for the pseudorandom generator (e.g., for reproducibility)
            nan: True returns array 'data_vars' as NaN. False returns default values.
            verbose: display values for variales, except 'data_vars' (see Returns below).
    Returns:
        Dictionary of variables for syncoord.utils.testdata with all the optional arguments plus
        an array 'data_vars'. Such array has dimensions [sections,points,axes,vars], where vars
        are frequency, phase shift, amplitude, vertical offset, irregularity, noise_strength.
    '''
    fps = kwargs.get('fps',30)
    durations_sections = kwargs.get('durations_sections',[4,4,4,4])
    n_points = kwargs.get('n_points',4)
    n_axes = kwargs.get('n_axes',2)
    seed = kwargs.get('seed',None)
    verbose = kwargs.get('verbose',False)

    n_sections = len(durations_sections)
    total_duration = sum(durations_sections)
    point_vars = np.empty((n_sections,n_points,n_axes,6)) # dim = [sections,points,axes,vars]
    point_vars[:] = np.nan

    data_vars = { 'fps':fps,'durations_sections':durations_sections,'total_duration':total_duration,
                   'n_points':n_points,'n_axes':n_axes,'point_vars':point_vars,'seed':seed }
    if verbose:
        print('sampling rate =',data_vars['fps'],'(fps or Hz)')
        print('duration of sections =',data_vars['durations_sections'],'(s)')
        print('total duration =',data_vars['total_duration'],'(s)')
        print('number of signals =', data_vars['n_points'])
        print('number of dimensions per signal =',data_vars['n_axes'])

    if ('nan' in kwargs) and kwargs['nan']: return data_vars

    # vars = frequency, phase_shift, amplitude, vertical_offset, irregularity, noise_strength
    # axis 1 ..........................................
    # section 0:
    point_vars[0,0,1] = 1, 0,   33, 100, 0,  0.5
    point_vars[0,1,1] = 1, 0,   37, 200, 0,  0.5
    point_vars[0,2,1] = 1, 0,    4, 400, 0,  0.5
    point_vars[0,3,1] = 1, 0,    5, 500, 0,  0.5
    # section 1:
    point_vars[1,0,1] = 1, 0,   45, 100, 0.1,  0.5
    point_vars[1,1,1] = 3, 0,   40, 200, 0.1,  0.5
    point_vars[1,2,1] = 1, 0,   18, 400, 0.1,  0.5
    point_vars[1,3,1] = 1, 0,   12, 500, 0.1,  0.5
    # section 2:
    point_vars[2,0,1] = 1, 0.2, 45, 100, 0.9,  0.5
    point_vars[2,1,1] = 1, 0.7, 40, 200, 0.7, 0.5
    point_vars[2,2,1] = 1, 0,   18, 400, 0.9,  0.5
    point_vars[2,3,1] = 1, 1.5, 12, 500, 0.8,  0.5
    # section 3:
    point_vars[3,0,1] = 1, 0,       45, 100, 0, 0.5
    point_vars[3,1,1] = 1, np.pi/2, 40, 200, 0, 0.5
    point_vars[3,2,1] = 1, 0,       18, 400, 0, 0.5
    point_vars[3,3,1] = 1, 2*np.pi, 12, 500, 0, 0.5
    # axis 0 ..........................................
    point_vars[:,:,0,:] = point_vars[:,:,1,:]
    point_vars[:,:,0,2] = point_vars[:,:,0,2] * 0.6
    point_vars[:,:,0,3] = 200

    return data_vars

def testdata(*args,**kwargs):
    '''
    Synthetic data for testing functions that measure synchronisation.
    Arguments can be the same keywords for function 'init_testdatavars,
    or a dictionary resulting from that function.
    If no arguments are given, default data will be produced with the function 'init_testdatavars'.
    Args:
        See documentation for syncoord.utils.init_testdatavars
    Returns:
        N-D array with dimensions [points,axes,frames]
    '''
    if args:
        kwargs = args[0]

    if not kwargs:
        kwargs = init_testdatavars()

    if kwargs:
        fps = kwargs.get('fps')
        durations_sections = kwargs.get('durations_sections')
        total_length = kwargs.get('total_duration')*fps
        n_points = kwargs.get('n_points')
        n_axes = kwargs.get('n_axes')
        seed = kwargs.get('seed')
        point_vars = kwargs.get('point_vars') # dim = [sections,points,axes,vars]
        # vars = frequency, phase_shift, amplitude, vertical_offset, irregularity, noise_strength

    test_data = np.empty((n_points,n_axes,total_length)) # dim = [points,axes,frames]
    wavargs = {}
    wavargs['sampling_frequency'] = fps
    wavargs['seed'] = seed
    i_start_section = 0
    for i_s,s in enumerate(point_vars):
        n_frames = durations_sections[i_s] * fps
        i_end_section = i_start_section + n_frames
        for i_p,p in enumerate(s):
            for i_ax,ax in enumerate(p):
                wavargs['frequency'] = ax[0]
                wavargs['phase_shift'] = ax[1]
                wavargs['amplitude'] = ax[2]
                wavargs['length'] = n_frames
                wavargs['vertical_offset'] = ax[3]
                wavargs['irregularity']  = ax[4]
                wavargs['noise_strength'] = ax[5]
                wavargs['seed'] += 1
                test_data[i_p,i_ax,i_start_section:i_end_section] = supersine(wavargs)
        i_start_section = i_end_section
    return test_data

def listfiles(path):
        '''
        Check if path is dir or file.
        If dir, return list of file names in folder.
        If file, return file name as list.
        Arg:
            path: str, path of folder or file
        Returns:
            fnames: list, file names with extensions
        '''
        if os.path.isdir(path):
            fnames = []
            _join = os.path.join
            _isfile = os.path.isfile
            for fn in os.listdir(path):
                if _isfile( _join(path, fn) ):
                    fnames.append(fn)
        elif os.path.isfile(path):
            fnames = [os.path.basename(path)]
        return fnames

def load_data( preproc_data, *props, annot_path=None, topdata_Name=None,
               max_n_files=None, print_info=True, **kwargs):
    '''
    Args:
        preproc_data (str,dict,numpy.ndarray):
            If str: Path to folder with parquet files for preprocesed data
                    (e.g., r"~/preprocessed"), or "make" to produce synthetic data with default
                    values (calls syncoord.utils.testdata). Data in the parquet file is a numerical
                    matrix with an index, and columns labelled as "point-number_dimension-label".
                    For example, for two points in two dimensions the column names
                    are: ['0_x','0_y','1_x','1_y']
            If dict: as returned by syncoord.utils.init_testdatavars
            If np.ndarray: as returned by syncoord.utils.testdata
        props (str,dict): Optional or ignored if preproc_data = "make"
            If str: Path for properties CSV file (e.g., r"~/properties.csv").
                    The properties file contain one or both of:
                    1) Arbitrary number of header rows as in dict, with comma-separated pairs of
                       property and value for all files.
                       Example:
                               fps,30
                               ndim,3
                    2) Properties for each file, where columns are properties and rows are files.
                       The first row of the table or after the headers, is for column names.
                       Example:
                               ID, fps
                               file_1,25
                               file_2,29.97
            If dict: The same properties (dict keys) will apply to all loaded files.
                     Properties:
                         props['fps'] (int): sample rate
                         props['ndim'] (int): number of dimensions. Default = 2
        Optional:
            annot_path (str): Path for annotations CSV file
                              (e.g., r"~/Pachelbel_Canon_in_D_String_Quartet.csv").
            topdata_Name (str,list): 'idx' for element index (e.g., anonymise),  list, or None
                                      for "Name" in annotation file.
            max_n_files (int): Number of files to extract from the beginning of annotations.
            print_info (bool): Print index, name and duration of data.
            **kwargs: passed to syncoord.utils.init_testdatavars if preproc_data = "make"
    Returns:
        topinfo (pandas.DataFrame): Properties and annotations (if they exist), and
                                    trimmed sections in frames if column "Sections" exist.
        dim_names (list): names of N-D array's dimensions*
                          (short, used e.g., to select data, in groupby, etc.)
        dim_labels (list): labels for N-D array's dimensions*
                           (less short, used e.g, as labels for visualisations)
        dimel_labels (list): labels for each element of N-D array's dimensions*.
        prep_data (dict): N-D numpy arrays containing preprocessed data.
        * dimensions = axes of the N-D Numpy array, where the rightmost is the fastest changing.
    '''
    properties = None
    ndim = 2
    if props[0] and isinstance(props[0][0],str):
        properties = pd.read_csv(props[0][0])
        if 'ID' not in properties.columns:
            params = [properties.columns.values.tolist()]
            if 'ID' in properties.iloc[:,0].values:
                i_start = np.where(properties.iloc[:,0] == 'ID')[0].item()
                properties.columns = properties.iloc[i_start]
                params.extend(properties[0:i_start].values.tolist())
                properties = properties.drop(properties.index[0:i_start])
                properties = properties.reset_index()
                properties = properties.drop('index',axis=1)
                properties.columns.name = ''
            else:
                params.extend(properties.values.tolist())
                properties = None
            del props
            props = (({},),)
            for k,v in params:
                if k == 'ndim': ndim = int(v)
                else: props[0][0][k] = v

    def make_topinfo_tdv(**kwargs):
        tdv = init_testdatavars(**kwargs)
        tsf = []
        csum = 0
        for d in tdv['durations_sections'][:-1]:
            csum += d
            tsf.append(csum * tdv['fps'])
        topinfo = pd.DataFrame( columns = ['ID','Name','fps','trimmed_sections_frames'],
                                data = [['test','Test Data',tdv['fps'],tsf]] )
        return topinfo, tdv

    if annot_path:
        # If annotations files exist, 'ID' of "annotations" will be the main order.
        annotations = pd.read_csv(annot_path)
        if max_n_files:
            annotations = annotations[:max_n_files]
        if properties:
            check_len_prop = properties.shape[0] != annotations.shape[0]
            assert check_len_prop, 'The lengths of properties and annotations are not equal.'
            topinfo = pd.merge(annotations,properties,on='ID')
        else:
            assert isinstance(props[0][0],dict), 'argument "props" should be dict or str'
            topinfo = annotations
            for k,v in props[0][0].items():
                check_v = isinstance(v,(int,float))
                assert check_v, '"props" should have only one value (int or float) per key'
                if k != 'ndim': topinfo[k] = v
        if 'Sections' in topinfo:
            topinfo['trimmed_sections_frames'] = trim_sections_to_frames(topinfo)
    elif properties: topinfo = properties

    prep_data = {}
    axes_labels = None
    if isinstance(preproc_data,str):
        if preproc_data == 'make':
            if not annot_path: topinfo, tdv = make_topinfo_tdv(**kwargs)
            prep_data[0] = testdata(tdv)
        else:
            for i in range(topinfo.shape[0]):
                ID = topinfo['ID'].iloc[i]
                top_df = pd.read_parquet(preproc_data + '/' + ID + '.parquet')
                top_df_ra = [ top_df.iloc[:,i_d::ndim].T for i_d in range(ndim) ]
                top_arr_nd = np.array(top_df_ra)
                top_arr_nd = np.transpose(top_arr_nd,(1,0,2))
                prep_data[i] = top_arr_nd
            axlbl = list(dict.fromkeys([s.split('_')[1] for s in top_df.columns.values]))
            axlbl.reverse()
            axes_labels = [f'${s}$' for s in axlbl]
    elif isinstance(preproc_data,dict):
        prep_data[0] = testdata(preproc_data)
        if not annot_path: topinfo, _ = make_topinfo_tdv(**kwargs)
    elif isinstance(preproc_data,np.ndarray):
        prep_data[0] = preproc_data
        if not annot_path: topinfo, _ = make_topinfo_tdv(**kwargs)

    if axes_labels is None:
        axlbl = ['x','y','z']
        axes_labels = [f'${axlbl[i]}$' for i in reversed(range(prep_data[0].shape[-2]))]
    dim_names = ['point','axis','frame']
    dim_labels = ['point','axes','time (frames)']
    dimel_labels = [['p. '+str(i) for i in range(prep_data[0].shape[dim_names.index('point')])],
                  axes_labels,None]
    if isinstance(topdata_Name,list):
        topinfo["Name"] = topdata_Name
    elif (topdata_Name=='idx') or ("Name" not in topinfo):
        topinfo["Name"] = [ str(c) for c in range(len(topinfo)) ]
    elif topdata_Name is None: pass
    else:
        raise Exception("invalid value for topdata_Name")
    d_keys = list(prep_data.keys())
    if print_info: print('index; key; Name; duration (s):')
    for i,k in enumerate(topinfo.index):
        if k != d_keys[i]:
            raise Exception("".join([f"ptdata.topinfo.index[{k}] doesn't match ",
                                     f"list(ptdata.data.keys())[{i}]"]))
        if print_info:
            this_length = prep_data[k][0].shape[-1]
            this_duration_lbl = frames_to_minsec_str(this_length,topinfo.loc[k,'fps'],ms=True)
            print(f'  {i}; {k}; {topinfo["Name"].iloc[i]}; {this_duration_lbl}')
    return topinfo, dim_names, dim_labels, dimel_labels, prep_data

def matlab_eng( addpaths=None, verbose=True ):
    '''
    Connects to Matab and returns a matlab.engine object.
    Optional args:
        addpaths: str or list, path(s) to add to Matlab, or None.
        verbose: True or False.
    '''
    if verbose: print('Connecting to Matlab...')
    import matlab.engine
    matlabeng = matlab.engine.start_matlab()
    if verbose: print('...connected to Matlab version',matlabeng.version())
    full_addpaths = ["../src" ]
    if addpaths:
        if isinstance(addpaths,str): full_addpaths.append(addpaths)
        elif isinstance(addpaths,list): full_addpaths.extend(addpaths)
    for p in full_addpaths:
        if p:
            pabs = os.path.abspath(p)
            matlabeng.addpath( matlabeng.genpath(p), nargout=0 )
    return matlabeng

def invexaxes(exaxes, s, d=None ):
    '''
    Turns selection of axes (dimensions) to exclude (except last), into the opposite.
    The last dimension will always be included.
    Useful to delete names or labels of dimensions that will not exist after a process.
    Args:
        exaxes: dimensions to exclude.
        s: shape of the array before processing.
        d: ndim (number of dimensions) of the array before processing.
    Returns:
        graxes: opposite to input except last.
        exaxes: as input; iterable if it is not.
    '''
    try:
        iter(exaxes)
        if isinstance(exaxes,tuple): exaxes = list(exaxes)
        exaxes.sort()
    except:
        if exaxes is None: exaxes = []
        else: exaxes = [exaxes]
    if (-1 in exaxes) or (s[-1] in exaxes):
        raise Exception('the last dimension cannot be excluded')
    if d is None: d = len(s)
    graxes = np.delete(np.arange(d),exaxes).tolist()
    return graxes, exaxes