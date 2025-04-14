'''Feature extraction from video files.'''

import os
import time
import subprocess
from datetime import timedelta

import numpy as np
import pandas as pd

from . import utils

def download( ID, mode, **kwargs):
    '''
    Download video from Youtube, get information, preview, and write text file 'properties.csv'
    with ID (identificaion code of the video), and fps (frames per second).
    Args:
        ID: str, identification code at the end of the video's Youtube page URL.
            The Youtube ID follows this string in the URL: "www.youtube.com/watch?v="
        mode: 'download', 'preview', or 'info'. They may be combined.
        Optional:
            video_folder: path to save the downloaded video file.
            prop_folder: path to save the properties file. Default is video_folder
            fn: name for the resulting file. If None, ID will be used.
            maxh: maximum height of video to download. Default = 720
            maxfps: maximum frame rate of video to download. Default = 60
            ext: file extension (encapsulation format) for resulting video.
            verbose
    Dependency:
        https://github.com/yt-dlp/yt-dlp
    '''
    #TO-DO: handle exceptions of YoutubeDL when downloading

    from yt_dlp import YoutubeDL
    from IPython.display import YouTubeVideo

    video_folder = kwargs.get('video_folder',None)
    prop_folder = kwargs.get('prop_folder',video_folder)
    fn = kwargs.get('fn',ID)
    maxh = kwargs.get('maxh',720)
    maxfps = kwargs.get('maxfps',60)
    ext = kwargs.get('ext','mp4')
    verbose = kwargs.get('verbose',False)

    if 'preview' in mode: display(YouTubeVideo(ID))

    yt_url = f'https://www.youtube.com/watch?v={ID}'

    if 'info' in mode:
        yt_info = subprocess.run(['yt-dlp','-F',yt_url],capture_output=True, text=True)
        print(yt_info.stdout)

    if 'download' in mode:

        assert video_folder, 'Kewyord argument "video_folder" is missing.'

        video_ffn_ne = video_folder + '/' + fn
        video_fn = f'{fn}.{ext}'
        if video_fn in os.listdir(video_folder):
            raise Exception(f'File {video_fn} already exists in video video_folder.')

        video_format_str = f'bv[height<=?{maxh}][fps<={maxfps}]+ba'
        ydl_opts = { 'format': video_format_str,
                     'outtmpl': video_ffn_ne,
                     'final_ext': ext,
                     'verbose': verbose,
                     'quiet':not verbose,
                     'no_warnings':not verbose,
                     'no-cache-dir': True,
                     'postprocessors': [{ 'key': 'FFmpegVideoRemuxer',
                                          'preferedformat': ext, }] }

        with YoutubeDL(ydl_opts) as ydl:
            ydl_info = ydl.extract_info(yt_url,download=True)
            fps = ydl_info['fps']

        props_df_new = pd.DataFrame([[ID,fps]], columns=["ID","fps"])
        if 'properties.csv' in os.listdir(prop_folder):
            props_df_old = pd.read_csv(prop_folder+r'\properties.csv')
            if ID in props_df_old.ID.values:
                raise Exception(f'ID = {ID} already exists in file "properties.csv".')
            props_df_new = pd.concat([props_df_old, props_df_new], axis=0)
        props_df_new.to_csv(prop_folder+'\properties.csv', index=False)

def getaudio( ffn_in, ffn_ne_out=None ):
    '''
    Extract audio from video file.
    Args:
        ffn_in: str, filename of input video file, with full or relative path stem.
        ffn_ne_out: str or None.
                    If str: filename of output audio file, with no extension,
                            with full or relative path stem.
                    If None: The audio file will have the same name and will be placed
                             in the same folder, as the input ideo file.
    Returns:
        str, extension (format) of the audio file
    Dependencies:
        ffmpeg and ffprobe installed in system (callable by command line)
    '''
    # TO-DO: handle exceptions
    audio_ext = ((subprocess.run([ "ffprobe","-v","error","-select_streams","a","-show_entries",
                                   "stream=codec_name","-of",
                                   "default=nokey=1:noprint_wrappers=1", ffn_in ],
                                   stdout=subprocess.PIPE).stdout).strip()).decode("utf-8")

    if not ffn_ne_out: ffn_out = os.path.splitext(ffn_in)[0] + '.' + audio_ext
    else: ffn_out = ffn_ne_out + '.' + audio_ext

    subprocess.run([ 'ffmpeg', '-y', '-loglevel', 'error', '-i', ffn_in, '-vn', 
                     '-acodec', 'copy', ffn_out ])
    return audio_ext

def setaudio( ffn_video_in, ffn_audio_in, ffn_video_out=None ):
    '''
    Put or replace audio in a video file.
    Args:
        ffn_video_in: Filename of input video file.*
        ffn_audio_in: Filename of input audio file.*
        ffn_video_out: Filename of output audio file.* If None, the fill will be saved in the same
                       folder as the input video file, with the same name plus label "+audio".
            * file names should include full or relative path
    Dependency:
        ffmpeg installed in system (callable by command line)
    '''
    # TO-DO: handle exceptions
    if not ffn_video_out:
        stem_video_in = os.path.splitext(ffn_video_in)[0]
        ext_video_in = os.path.splitext(ffn_video_in)[1]
        ffn_video_out = stem_video_in + '+audio' + ext_video_in

    subprocess.run([ 'ffmpeg', '-y', '-loglevel', 'error', '-i', ffn_video_in,
                     '-i', ffn_audio_in, '-c', 'copy', ffn_video_out ])

def posetrack( video_in_path, json_path, AlphaPose_path, **kwargs ):
    '''
    Detect and track human pose in video files.
    Note:
        Thsi function overrides detector parameters in file AlphaPose\detector\yolo_cfg.py
        See documentation for more information (links at the bottom).
    Args:
        video_in_path: str, path for input video file or folder with input video files.
        json_path: str, path of folder for resulting json tracking files.
        AlphaPose_path: str, path of folder where AlphaPose code is.
        Optional:
            video_out_path: str, path of folder for resulting tracking video files
                            with superimposed skeletons. None = Don't make video.
            trim_range = list, [start,end] in seconds or 'end'.
            log_path: str, path of folder for log file.
            skip_done: skip if corresponding json file exists in json_path, default = False
            idim: int or list. Size of the detection network, multiple of 32.
            thre: float or list. NMS threshold for detection in (0...1]
            conf: float or list. Confidence threshold for detection in (0...1]
            parlbl: bool, add [idim,thre,conf] to the names of the resulting files.
            suffix: str, label to be added to the names of the resulting files.
            audio: bool, extract audio from tracking video and add to AlphaPose video files.
                   Audio files will be saved in video_in_path, video files with added audio
                   will be saved in video_out_path.
            sp: bool. Run on a single process. Forcefully True if operating system is Windows.
            flip: bool. Enable flip testing. It might improve accuracy.
            model_paths: dict;
                model_paths['model']: str, path for pretrained model.
                model_paths['config']: str, path pretrained model's configuration file.
            verbosity: 0, 1, or 2. Only for notebook view, otherwise full verbosity.
    Dependencies:
        AlphaPose fork: https://github.com/juigmend/AlphaPose
        ffmpeg installed in system (callable by command line)
    Documentation:
        https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/run.md
        https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/speed_up.md
        https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
    '''
    video_out_path = kwargs.get('video_out_path',None)
    trim_range = kwargs.get('trim_range',None)
    log_path = kwargs.get('log_path',None)
    skip_done = kwargs.get('skip_done',True)
    idim = kwargs.get('idim',608)
    thre = kwargs.get('thre',0.6)
    conf = kwargs.get('conf',0.1)
    parlbl = kwargs.get('parlbl',False)
    suffix = kwargs.get('suffix','')
    audio = kwargs.get('audio',True)
    sp = kwargs.get('sp',False)
    flip = kwargs.get('flip',False)
    model_path = kwargs.get('model',AlphaPose_path
                            + r'\pretrained_models\fast_421_res152_256x192.pth')
    model_config_path = kwargs.get('config',AlphaPose_path
                                    + r'\configs\coco\resnet\256x192_res152_lr1e-3_1x-duc.yaml')
    verbosity = kwargs.get('verbosity',1)

    if video_in_path: video_in_path = os.path.abspath(video_in_path)
    if json_path: json_path = os.path.abspath(json_path)
    if video_out_path: video_out_path = os.path.abspath(video_out_path)
    if log_path: log_path = os.path.abspath(log_path)

    if video_out_path: save_video_cmd = ['--visoutdir',video_out_path,'--save_video']
    else: save_video_cmd = ['','','']

    do_trim_video = not(not trim_range \
                        or ((trim_range[0] == 0) & (trim_range[1] == 'end')))
    if do_trim_video:
        check_trim = isinstance(trim_range[0],(float,int)) \
                     and (isinstance(trim_range[1],(float,int)) or (trim_range[1] == 'end'))
        assert check_trim, 'Trim range values are incorrect.'
        trim_lbl = f'_{trim_range[0]}-{trim_range[1]}'
        video_out_folder = video_in_path + r'\trimmed'
        if not os.path.exists(video_out_folder): os.makedirs(video_out_folder)
    else:
        trim_lbl = ''
        video_out_folder = video_in_path

    if skip_done:
        json_saved_fn = []
        for fn in os.listdir(json_path):
            ffn_json = os.path.join(json_path, fn)
            if os.path.isfile(ffn_json): json_saved_fn.append( fn )

    if suffix: suffix_cmd = ['--suffix',suffix]
    else: suffix_cmd = ''

    if audio:
        audio_out_folder = os.path.join(video_out_folder,'audio')
        if not os.path.exists(audio_out_folder): os.makedirs(audio_out_folder)

    if sp: sp_cmd = '--sp'
    else: sp_cmd = ''

    if flip: flip_cmd = '--flip'
    else: flip_cmd = ''

    fnames = utils.listfiles(video_in_path)

    def one_posetrack_( idim_, thre_, conf_ ):

        for fn in fnames:
            ffn = os.path.join(video_in_path,fn)
            split_fn = os.path.splitext(fn)
            fn_ne = split_fn[0] + trim_lbl
            json_fn = f'AlphaPose_{fn_ne}{suffix}.json'
            new_file = True
            if skip_done: new_file = json_fn not in json_saved_fn
            if parlbl or log_path or verbosity:
                par_str_long = f'idim = {idim_}; thre = {thre_}; conf = {conf_}'
            if verbosity:
                print(f'{fn} :\n{par_str_long}')

            if (not skip_done) or new_file:

                if parlbl or log_path: par_str_short = f'[{idim_},{thre_},{conf_}]'
                if parlbl:
                    suffix_cmd[0] = '--suffix'
                    suffix_cmd[1] = f'_{par_str_short}{suffix}'
                if verbosity: print('Processing...',end=' ')
                if verbosity or log_path: tic = time.time()
                if log_path: tracking_log_txt = [f'{fn}\n{par_str_long}\n']

                # Trim video:
                if do_trim_video:
                    if trim_range[1] == 'end':
                        vsp_args = [ "ffprobe", "-v", "error", "-show_entries",
                                     "format=duration", "-of",
                                     "default=noprint_wrappers=1:nokey=1", ffn ]
                        video_duration = subprocess.run( vsp_args,
                                                         stdout=subprocess.PIPE,
                                                         stderr=subprocess.STDOUT)
                        # overshoot to ensure its the end of the video:
                        trim_end = int(float(video_duration.stdout)) + 1
                    else: trim_end = trim_range[1]
                    video_to_track_ffn = ''.join([ video_out_folder, '\\', split_fn[0],
                                                   trim_lbl, split_fn[1] ])
                    if not os.path.isfile(video_to_track_ffn):
                        ffmpeg_cmd = [ 'ffmpeg','-y','-loglevel','error','-i',ffn,'-ss',
                                        f'{trim_range[0]}','-to',f'{trim_end}',
                                        '-q:a',str(0),'-q:v',str(0),video_to_track_ffn ]
                        subprocess.run( ffmpeg_cmd )
                else:
                    video_to_track_ffn = ffn

                # Get audio from input video:
                if video_out_path and audio:
                    audio_ffn = os.path.join(audio_out_folder,fn_ne)
                    audio_ext = getaudio( video_to_track_ffn, audio_ffn )

                # AlphaPose:
                AlphaPose_cmd = [ 'cd',AlphaPose_path,'&&','python',r'scripts\demo_inference.py',
                                  '--sp','--video',video_to_track_ffn,'--jsonoutdir',json_path,
                                   save_video_cmd[0],save_video_cmd[1],save_video_cmd[2],
                                  '--checkpoint',model_path,'--cfg',model_config_path,
                                  '--pose_track',suffix_cmd[0],suffix_cmd[1],'--vis_fast',
                                  '--param',str(idim_),str(thre_),str(conf_),sp_cmd,flip_cmd ]
                AlphaPose_cmd = [s for s in AlphaPose_cmd if s != '']

                def run_AP_(cmd):
                    clear_line = True
                    AP_out = subprocess.Popen( cmd, shell=True,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True)
                    for l in iter(AP_out.stdout.readline,''):
                        if l:
                            if clear_line and verbosity:
                                for _ in range(14): print('\b',end='',flush=True)
                                print()
                                clear_line = False
                            yield l
                    AP_out.stdout.close()
                    rc = AP_out.wait()
                    if rc: raise subprocess.CalledProcessError(rc, cmd)
                    if AP_out.stderr: print('\nstderr:\n',AP_out.stderr)

                if verbosity==2:
                    for l in run_AP_(AlphaPose_cmd): print(l)
                else:
                    subprocess.run( AlphaPose_cmd, shell=True )
                    if verbosity:
                        for _ in range(14): print('\b',end='',flush=True)
                        print('')

                if verbosity or log_path:
                    toc = round(time.time() - tic,3)
                    toc_str = f"computing time = {str(timedelta(seconds=toc))[:-3]} (H:M:S)\n"
                    if verbosity: print(toc_str,'\n')

                # Set audio to output video:
                if video_out_path and audio:
                    tracked_out_fn = 'AlphaPose_' \
                                      + split_fn[0] + trim_lbl + suffix_cmd[1] + split_fn[1]
                    tracked_video_ffn = os.path.join(save_video_cmd[1],tracked_out_fn)
                    setaudio( tracked_video_ffn, audio_ffn + '.' + audio_ext )

                # save log:
                if log_path:
                    tracking_log_txt.append(toc_str)
                    txtlog_ffn = log_path + '\\' + 'posetrack_log.txt'
                    tracking_log_txt.append('\n')
                    with open(txtlog_ffn, 'a') as output:
                        for t in tracking_log_txt:
                            output.write(t)

            elif verbosity: print('skipped')

    if not isinstance(idim,list): idim = [idim]
    if not isinstance(thre,list): thre = [thre]
    if not isinstance(conf,list): conf = [conf]

    for idim_ in idim:
        for thre_ in thre:
            for conf_ in conf:
                one_posetrack_( idim_, thre_, conf_ )

def poseprep( json_path, savepaths, vis={}, **kwargs ):
    '''
    Pre-process AlphaPose's tracking results.
    Currently tested for extraction of one point per individual.
    Args:
        json_path: str, path of folder for input json AlphaPose tracking files.
        savepaths: dict, of str or None.
                savepaths['parquet']: path to folder for pre-processed data parquet files.
                Optional:
                    savepaths['rawfig']: path to folder for raw data figures.
                    savepaths['prepfig']: path to folder for pre-processed data figures.
                    savepaths['log']: path to folder for pre-processing log file.
        Optional:
            vis: dict, visualisation options.
                vis['show']: bool, show visualisation (independent of saving)
                vis['markersize']: scalar, marker size for raw data plots.
                vis['lwraw']: scalar, line width for raw data plots.
                vis['lwprep']: scalar, line width for pre-processed data plots.
        **kwargs:
            keypoints: list. Default =[0,1] ([x1,y1] for "Nose", assuming COCO format).
            n_indiv: int, expected number of individuals to be tracked, None for automatic.
            skip_done: bool, skip if corresponding preprocessed parquet file exists.
            suffix: str, label to be added to the names of the resulting files.
            trange: list or str. Time-range selection.
                    If list: [start,end] (frames)
                    If str: 'all'
            drdim: None, 'all', int, or list of dimensions to apply disjoint ranges
                   for tracked individuals. Works only if keypoint trajectories don't overlap.
            verbose: bool.
    Documentation:
        https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md
    '''
    assert isinstance(savepaths,dict), 'Keyword argument "savepaths" should be dict.'
    assert savepaths['parquet'], 'savepaths["parquet"] not in input.'
    preproc_path = savepaths['parquet'] # TO-DO: save to numpy file
    rawfig_path = savepaths.get('rawfig',None)
    prepfig_path = savepaths.get('prepfig',None)
    log_path = savepaths.get('log',None)

    assert isinstance(vis,dict), 'Argument "vis" should be dict.'
    vis = {'show':True,'markersize':0.8,'lwraw':4,'lwprep':2,**vis}
    keypoints = kwargs.get('keypoints',[0,1])
    if len(keypoints) > 2:
        raise Exception(''.join([ f'Currently only one point with two dimensions (x,y) \
                                    are allowed, but instead got this: {keypoints}' ]))
    n_indiv = kwargs.get('n_indiv',None)
    skip_done = kwargs.get('skip_done',True)
    suffix = kwargs.get('suffix',None)
    trange = kwargs.get('trange',None)
    drdim = kwargs.get('drdim',None)
    verbose = kwargs.get('verbose',True)

    if drdim:
        from sklearn.cluster import HDBSCAN
    if vis['show'] or rawfig_path or prepfig_path:
        import matplotlib.pyplot as plt

    DIM_LABELS = ['x','y']
    if isinstance(drdim,int): drdim = [drdim]
    elif isinstance(drdim,str): drdim = list(range(len(DIM_LABELS)))

    parquet_fnames = utils.listfiles(savepaths['parquet'])
    json_fnames = utils.listfiles(json_path)
    for json_fn in json_fnames:

        fn_ne = os.path.splitext(json_fn)[0]
        if verbose: print(f'{json_fn} :',end=' ')

        parquet_fn = f'{fn_ne}.parquet'
        new_file = True
        if skip_done: new_file = parquet_fn not in parquet_fnames

        if (skip_done is None) or new_file:

            if verbose: print('processing...')

            # Load data from JSON file produced by AlphaPose:
            data_raw_df = pd.read_json(json_path + '\\' + json_fn)
            if n_indiv is None: n_persons = data_raw_df.idx.max()
            else: n_persons = n_indiv
            persons_range = range(1,n_persons+1)

            # Reduce by removing unnecessary data:
            data_red_df = data_raw_df.drop(['category_id','keypoints','score','box'],axis=1)
            for lbl,i in zip(DIM_LABELS,keypoints):
                data_red_df[lbl] = data_raw_df.keypoints.str[i]
            data_red_df.image_id = data_red_df.image_id.str.split('.').str[0].astype(int)

            # Inspect and make plot of raw data:
            if drdim: limits = []
            if vis['show'] or rawfig_path or log_path or drdim:
                if log_path: prep_log_txt = [fn_ne + '\n']
                if (trange is None) or (trange == 'all'):
                    t_loc = [0,data_red_df.image_id.max()]
                else:
                    t_loc = trange
                n_series = len(keypoints)
                series_range = range(n_series)
                for i_s in range(n_series):
                    if vis['show'] or rawfig_path: plt.subplot(n_series,1,i_s+1)
                    n_frames = []
                    legend = []
                    for i_p in persons_range:
                        slice_sel = (   (data_red_df.idx == i_p)
                                      & (data_red_df.image_id >= t_loc[0])
                                      & (data_red_df.image_id <  t_loc[1]) )
                        data_red_slice_df = data_red_df[DIM_LABELS[i_s]][slice_sel]
                        if vis['show'] or rawfig_path:
                            data_red_slice_df.plot(linewidth=vis['lwraw'])
                        n_frames.append(len(data_red_slice_df))

                    if vis['show'] or rawfig_path or drdim:
                        this_series = data_red_df[DIM_LABELS[i_s]].sort_values()
                    if vis['show'] or rawfig_path:
                        this_series.plot( marker='.', linestyle='none',
                                          markersize=vis['markersize'], color='k')
                    plot_hlines = False
                    if drdim and (i_s in drdim):
                        mcs = int(len(this_series)/(n_persons*2))
                        clustering = HDBSCAN( min_cluster_size=mcs, store_centers="centroid",
                                              metric="cityblock" )
                        clustering.fit(np.reshape(this_series, (-1, 1)))
                        centroids = np.squeeze(clustering.centroids_)
                        if centroids.size == n_persons:
                            these_limits = centroids[:-1] + np.diff(centroids)/2
                            these_limits = np.insert( these_limits, [0,len(centroids)-1],
                                                      [0,this_series.max()] )
                            limits.append(these_limits)
                            plot_hlines = True
                        else:
                            limits.append(None)
                            print( ''.join([ 'Warning: no disjoint ranges for axis',
                                             f'{i_s}', ' (', f'{DIM_LABELS[i_s]}', ')' ]))

                        if plot_hlines: plt.hlines( these_limits[1:-1], 0, len(this_series),
                                                    linestyles='dashed',
                                                    colors='tab:gray', linewidths=0.8 )

                    plt.ylabel(DIM_LABELS[i_s])
                    if i_s == 0:
                        plt.legend( list(persons_range)+['all'],loc='upper right',
                                    bbox_to_anchor=(1.2, 1.02) )

                    if log_path:
                        mean_persons = sum(n_frames)/n_persons
                        for p in n_frames:
                            if p != mean_persons:
                                warning_frames = ''.join([ 'inconsistent frame count in '
                                                          f'{DIM_LABELS[i_s]} {tuple(n_frames)}' ])
                                prep_log_txt.append( warning_frames+'\n' )
                                if verbose: print('Warning:',warning_frames)
                                break
                if rawfig_path or vis['show']:
                    plt.gcf().suptitle(fn_ne+'\nRaw Data')
                    plt.gcf().supxlabel('stacked frames (as in json file)')
                    plt.tight_layout()
                    if rawfig_path:
                        fig_ffn = rawfig_path + '\\' + fn_ne + '_RAW.png'
                        plt.savefig(fig_ffn)
                    if vis['show']: plt.show()
                    else: plt.close(plt.gcf())
                if log_path or verbose:
                    if (data_red_df.idx.max()) != n_persons:
                        warning_idx = 'more idx than number of people in raw data'
                        if verbose: print('Warning:',warning_idx)
                        if log_path: prep_log_txt.append(warning_idx+'\n')

            # Rearrange such that each row is a frame (image_id):
            data_rar_df = pd.DataFrame( list(range(data_red_df.image_id.max() + 1)) ,
                                        columns=['image_id'] )
            for i_p in persons_range:
                red_df_cols = ['image_id'] + DIM_LABELS
                red_df_idx = data_red_df.idx == i_p
                data_rar_df = data_rar_df.merge( data_red_df[red_df_cols][(red_df_idx)],
                                                 on='image_id', how='left',
                                                 suffixes=(f'_{i_p-1}',f'_{i_p}') )
            data_rar_df = data_rar_df.drop(['image_id'],axis=1)

            # Re-order and re-label columns in order from left to right as they appear in the image:
            # It is assumed that the persons don't relocate (e.g. they are sitting or standing in
            # one place). Indices are set to start at 0 to be consistent with Python indexing.
            new_order_x = [ x for x in data_rar_df.iloc[:,::2].median().sort_values().index]
            new_order_y = [ y.replace(DIM_LABELS[0],DIM_LABELS[1]) for y in new_order_x ]
            new_order_xy = []
            new_order_lbl = []
            i_c = 0
            for x,y in zip(new_order_x,new_order_y):
                new_order_xy.append(x)
                new_order_xy.append(y)
                new_order_lbl.append(f'{i_c}_{DIM_LABELS[0]}')
                new_order_lbl.append(f'{i_c}_{DIM_LABELS[1]}')
                i_c += 1
            data_rar_df = data_rar_df.reindex(new_order_xy, axis=1)
            data_rar_df.columns = new_order_lbl

            # Apply disjoint ranges:
            if drdim:
                i_lims = 0
                for d in drdim:
                    if isinstance(limits[i_lims],np.ndarray):
                        i_col = 0
                        for col_lbl in data_rar_df:
                            if (col_lbl[ col_lbl.index('_')+1: ]) == DIM_LABELS[d]:
                                lim_lo = data_rar_df[col_lbl] > limits[i_lims][i_col]
                                lim_hi = data_rar_df[col_lbl] <= limits[i_lims][i_col+1]
                                mask = lim_lo & lim_hi
                                data_rar_df[col_lbl] = data_rar_df.loc[mask,col_lbl]
                                i_col += 1
                    i_lims += 1

            # Fill missing data:
            found_nan = data_rar_df.isnull().values.any()
            if found_nan:
                data_rar_df = data_rar_df.interpolate(limit_direction='both',method='cubicspline')
                if log_path or verbose:
                    warning_interp = 'missing raw data have been interpolated'
                    if verbose:
                        print('Warning:',warning_interp)
                    if log_path:
                        prep_log_txt.append(warning_interp+'\n')

            # save log:
            if log_path:
                txtlog_ffn = log_path + '\\' + 'poseprep_log.txt'
                prep_log_txt.append('\n')
                with open(txtlog_ffn, 'a') as output:
                    for t in prep_log_txt:
                        output.write(t)

            # Make plot of pre-processed data:
            if vis['show'] or prepfig_path:
                if (trange is None) or (trange == 'all'):
                    t_loc = [0,data_rar_df.index.max()]
                else:
                    t_loc = trange
                for i_s in series_range:
                    plt.subplot(n_series,1,i_s+1)
                    legend = []
                    names_cols = [ f'{n}_{DIM_LABELS[i_s]}' for n in range(n_persons)]
                    for nc in names_cols:
                        data_rar_slice_df = data_rar_df[nc].iloc[ t_loc[0] : t_loc[1] ]
                        data_rar_slice_df.plot(linewidth=vis['lwprep'])

                        legend.append(nc.split('_')[0])
                    plt.ylabel(DIM_LABELS[i_s])
                    if i_s == 0:
                        plt.legend(legend,loc='upper right', bbox_to_anchor=(1.2, 1.02))
                plt.suptitle(fn_ne+'\nPre-processed Data')
                plt.xlabel('time (video frames)')
                plt.tight_layout()
                if prepfig_path:
                    fig_ffn = prepfig_path + '\\' + fn_ne + '_PREP.png'
                    plt.savefig(fig_ffn)
                if vis['show']: plt.show()
                else: plt.close(plt.gcf())

            # Write pre-processed data to a file:
            if savepaths['parquet']:
                parquet_ffn = preproc_path + '\\' + parquet_fn
                data_rar_df.to_parquet(parquet_ffn)

            if verbose: print('done')

        else:
            if verbose: print('skipped')