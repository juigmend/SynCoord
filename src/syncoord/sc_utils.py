'''Miscellaneous functionality.'''

import numpy as np
import pandas as pd

def halt(*arg):
    if not arg: arg = ('*** halt ***',)
    raise Exception(arg[0])

def frames_to_minsec_str(frames,fps):
    '''
    Convert frames (int) into a string with format "minutes:seconds".
    '''
    mins, sec = divmod(round(frames/fps), 60)
    if sec == 0: lbl = f'{mins}:00'
    elif sec < 10: lbl = f'{mins}:0{sec}'
    else: lbl = f'{mins}:{sec}'
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
        DataFrame with columns 'Start', 'Sections' and 'fps'.
        help(minsec_str_to_frames) for the format of cells in column 'Sections'.
    Returns:
        List with processed values.
    '''
    ts_f = []
    for i in topinfo.index:
        s_f = minsec_str_to_frames( topinfo['Sections'].iloc[i] , topinfo['fps'].iloc[i] )
        offset_s = topinfo['Start'].iloc[i]
        if np.isnan(offset_s): offset_s = 0
        offset_f = offset_s * topinfo['fps'].iloc[i]
        ts_f.append( [ round(f - offset_f) for f in s_f ] )
    return ts_f

def load_data( preproc_data_folder, prop_path, annot_path=None, topdata_Name='idx',
               max_n_files=None, print_durations=True ):
    '''
    Args:
        preproc_data_folder: folder of preprocessed position data parquet files
                             (e.g., r"~\preprocessed").
        prop_path: path for properties CSV file (e.g., r"~\properties.csv").
        Optional:
            annot_path: path for annotations CSV file
                        (e.g., r"~\Pachelbel_Canon_in_D_String_Quartet.csv").
            topdata_Name: 'idx' for element index (e.g., anonymise),  list, or empty for "Name" in
                           annotation file.
            max_n_files: number of files to extract from the beginning of annotations.
                         None or Scalar.
            print_durations: print durations of data. True or False.
    Returns:
        position_data: dictionary of multidimensional numpy arrays containing preprocessed data
        dim_names: names of N-D array's dimensions*
                   (short, used e.g., to select data, in groupby, etc.)
        dim_labels: labels for N-D array's dimensions*
                    (less short, used e.g, as labels for visualisations)
        dimel_labels: labels for each element of N-D array's dimensions*.
        topinfo: DataFrame with properties and annotations (if they exist)
                 with trimmed sections in frames.
        * dimensions = axes of the N-D Numpy array, where the rightmost is the fastest changing.
    '''
    properties = pd.read_csv(prop_path)

    if annot_path: # If annotations exist, 'ID' of "annotations" will be the main order.
        annotations = pd.read_csv(annot_path)
        if max_n_files:
            annotations = annotations[:max_n_files]
        if properties.shape[0] != annotations.shape[0]:
            raise Exception('The lengths of properties and annotations are not equal.')
        topinfo = pd.merge(annotations,properties,on='ID')
        topinfo['trimmed_sections_frames'] = trim_sections_to_frames(topinfo)
    else: topinfo = properties

    position_data = {}
    for i in range(topinfo.shape[0]):
        ID = topinfo['ID'].iloc[i]
        top_df = pd.read_parquet(preproc_data_folder + '\\' + ID + '.parquet')
        top_ndarr = np.array([ top_df.iloc[:,1::2].T , top_df.iloc[:,::2].T ])
        top_ndarr = np.transpose(top_ndarr,(1,0,2))
        position_data[i] = top_ndarr
    dim_names = ['point','axis','frame']
    dim_labels = ['point','axes','time (frames)']
    dimel_labels = [['p. '+str(i) for i in range(top_ndarr.shape[dim_names.index('point')])],
                  ['$y$','$x$'],None]
    if isinstance(topdata_Name,list):
        topinfo["Name"] = topdata_Name
    elif (topdata_Name=='idx') or ("Name" not in topinfo):
        topinfo["Name"] = [ str(c) for c in range(len(topinfo)) ]
    else:
        raise Exception("invalid value for topdata_Name")
    if print_durations:
        print('ID; Name; Duration:\n')
        for i in range(topinfo.shape[0]):
            this_length = position_data[i][0].shape[-1]
            this_duration_lbl = frames_to_minsec_str(this_length,topinfo['fps'].iloc[i])
            ID = topinfo.index[i]
            print(f'{ID}; {topinfo["Name"].iloc[i]};',this_duration_lbl)
    return position_data, dim_names, dim_labels, dimel_labels, topinfo
