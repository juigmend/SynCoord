'''Feature extraction from video files.'''

import os
import sys
import time
import subprocess
from datetime import timedelta

import numpy as np
import pandas as pd

from . import utils

def trop( path_in, path_out, **kwargs):
    '''
    Trim and crop video files.
    Args:
        path_in: input video file or folder
        path_out: output video file or folder
        Optional:
        (either trim or crop are optional)
            trim (list): [start,end] in seconds.
            crop (str): 'width:height:right:down' *
                Example: '400:200:50:10' results in 400 px. (pixels) of width, 200 px. of height,
                         at 50 px. right and 10 px. down from the top-left corner.
            skip_done (bool): Skip if output file exists. Default = False
            lbl (str): suffix for the output file's name or '--auto'. Default = None
    * Documentation: https://ffmpeg.org/ffmpeg-filters.html#crop
    '''
    trim = kwargs.get('trim',None)
    crop = kwargs.get('crop',None)
    skip_done = kwargs.get('skip_done',False)
    lbl = kwargs.get('lbl',None)

    assert (trim or crop), 'Check that at least one argument "trim" or "crop" has a value.'
    assert os.path.abspath(path_in) != os.path.abspath(path_out), 'path_in is the same as path_out'

    if os.path.isfile(path_in): video_in_path = os.path.dirname(path_in)
    else: video_in_path = path_in
    is_path_out_dir = os.path.isdir(path_out)

    if not lbl: lbl = ''
    if (lbl=='--auto'):
        if trim: trim_lbl = f'_{trim[0]}-{trim[1]}'
        else: trim_lbl = ''
        if crop:
            crop_lbl = crop.replace(':','-')
            crop_lbl = f'_[{crop_lbl}]'
        else: crop_lbl = ''
        lbl = f'{trim_lbl}{crop_lbl}'

    sp_cmd_a = ['ffmpeg', '-y', '-loglevel','error', '-i']
    sp_cmd_b = []
    if trim: sp_cmd_b += ['-ss', str(trim[0]), '-to', str(trim[1])]
    if crop: sp_cmd_b += ['-vf',f'crop={crop}']
    sp_cmd_b += ['-acodec', 'copy']

    for fn in utils.listfiles(path_in):
        ffn_in = os.path.join(video_in_path,fn)
        if is_path_out_dir:
            bname_split = os.path.splitext(fn)
            bname_lbl = bname_split[0] + lbl + bname_split[1]
            ffn_out = os.path.join(path_out, bname_lbl)
        else:
            ffn_out = path_out

        if skip_done: run_sp = not os.path.exists(ffn_out)
        else: run_sp = True

        if run_sp:
            sp_cmd = sp_cmd_a + [ffn_in] + sp_cmd_b + [ffn_out]
            subprocess.run(sp_cmd)
            assert os.path.exists(ffn_out), 'File not saved. Check that the output folder exists.'

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

    video_folder = kwargs.get('video_folder',None)
    prop_folder = kwargs.get('prop_folder',video_folder)
    fn = kwargs.get('fn',ID)
    maxh = kwargs.get('maxh',720)
    maxfps = kwargs.get('maxfps',60)
    ext = kwargs.get('ext','mp4')
    verbose = kwargs.get('verbose',False)

    if 'preview' in mode:
        from IPython.display import YouTubeVideo
        display(YouTubeVideo(ID))

    yt_url = f'https://www.youtube.com/watch?v={ID}'

    if 'info' in mode:
        yt_info = subprocess.run(['yt-dlp','-F',yt_url],capture_output=True, text=True)
        print(yt_info.stdout)

    if 'download' in mode:

        from yt_dlp import YoutubeDL
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
            props_df_old = pd.read_csv(prop_folder+'/properties.csv')
            if ID in props_df_old.ID.values:
                raise Exception(f'ID = {ID} already exists in file "properties.csv".')
            props_df_new = pd.concat([props_df_old, props_df_new], axis=0)
        props_df_new.to_csv(prop_folder+'/properties.csv', index=False)

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
        Optional argments idim, nms and conf override corresponding detector parameters in
        file AlphaPose/detector/yolo_cfg.py. See documentation (links at the bottom).
        This function has been tested using:
            - Windows 11, CPU
            - SLURM, CPU and GPU
    Args:
        video_in_path: str, absolute path for input video file or folder with input video files.
        json_path: str, absolute path of folder for resulting json tracking files.
        AlphaPose_path: str, absolute path of folder where AlphaPose code is.
        Optional:
            video_out_path: str. Path of folder for resulting tracking video files
                            with superimposed skeletons. None = Don't make video.
            trim_range = list. [start,end] in seconds or 'end'.
            log_path: str. Path of folder for log file.
            skip_done: skip if corresponding json file exists in json_path, default = False
            idim: int or list. Size of the detection network, multiple of 32.
            nms: float or list. NMS threshold for detection in (0...1]
            conf: float or list. Confidence threshold for detection in (0...1]
            parlbl: bool. Add [idim,nms,conf] to the names of the resulting files.
            suffix: str. Label to be added to the names of the resulting files.
            audio: bool. Extract audio from tracking video and add to AlphaPose video files.
                   Audio files will be saved in video_in_path, video files with added audio
                   will be saved in video_out_path.
            sp: bool. Run on a single process. Forcefully True if operating system is Windows.
            gpus: str. Index of CUDA device. Comma to use several, e.g. "0,1,2,3".
                       Use "-1" for cpu only. Default="0"
            program: str or None (default).
                     If str: 'inference' (module) or 'demo_inference' (script)
                              The former prints nicely on a Python IDE (e.g., Jupyter),
                              but may not work with GPU.
                     If None and gpus = "-1": use module AlphaPose/inference.py
                     Else: run script AlphaPose/scripts/demo_inference.py
            flip: bool. Enable flip testing. It might improve accuracy.
            detector: str. See documentation for available detectors. Default = 'yolo'
            model: str. Path for pretrained model (A.K.A. checkpoint).
            config: str. Path for pretrained model's configuration file.
            vis_fast: bool. Simpler and faster visualisation. Default = True
            verbosity: 0 (minimal), 1 (progress bar), or 2 (full). Default = 1
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
    nms = kwargs.get('nms',0.6)
    conf = kwargs.get('conf',0.1)
    parlbl = kwargs.get('parlbl',False)
    suffix = kwargs.get('suffix','')
    audio = kwargs.get('audio',True)
    sp = kwargs.get('sp',False)
    gpus = kwargs.get('gpus','0')
    program = kwargs.get('program',None)
    if program is None:
        if len(gpus) > 2: program = 'demo_inference'
        elif int(gpus) >= 0:  program = 'demo_inference'
        elif int(gpus) == -1: program = 'inference'
        else: raise Exception('invalid value for argument "gpus"')
    flip = kwargs.get('flip',False)
    detector = kwargs.get('detector','yolo')
    model_path = kwargs.get('model',AlphaPose_path
                            + '/pretrained_models/fast_421_res152_256x192.pth')
    model_config_path = kwargs.get('config',AlphaPose_path
                                    + '/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml')
    vis_fast = kwargs.get('vis_fast',True)
    verbosity = kwargs.get('verbosity',1)

    if video_in_path: video_in_path = os.path.abspath(video_in_path)
    if json_path: json_path = os.path.abspath(json_path)
    if video_out_path: video_out_path = os.path.abspath(video_out_path)
    if log_path: log_path = os.path.abspath(log_path)

    if video_out_path: save_video = True
    else: save_video = False

    do_trim_video = not(not trim_range \
                        or ((trim_range[0] == 0) & (trim_range[1] == 'end')))
    if do_trim_video:
        check_trim = isinstance(trim_range[0],(float,int)) \
                     and (isinstance(trim_range[1],(float,int)) or (trim_range[1] == 'end'))
        assert check_trim, 'Trim range values are incorrect.'
        trim_lbl = f'_{round(trim_range[0],2)}-{round(trim_range[1],2)}'
        video_out_folder = video_in_path + '/trimmed'
        if not os.path.exists(video_out_folder): os.makedirs(video_out_folder)
    else:
        trim_lbl = ''
        video_out_folder = video_in_path

    if skip_done:
        json_saved_fn = []
        for fn in os.listdir(json_path):
            ffn_json = os.path.join(json_path, fn)
            if os.path.isfile(ffn_json): json_saved_fn.append( fn )

    if audio:
        audio_out_folder = os.path.join(video_out_folder,'audio')
        if not os.path.exists(audio_out_folder): os.makedirs(audio_out_folder)

    fnames = utils.listfiles(video_in_path)

    cwd = os.getcwd()
    sys.path.append(AlphaPose_path)
    os.chdir(AlphaPose_path)

    if program == 'inference':

        import inference
        def _run_AP(video_to_track_ffn, idim_, nms_, conf_, suffix_str):
            alphapose_argdict = { 'video' : video_to_track_ffn,
                                  'jsonoutdir' : json_path,
                                  'visoutdir' : video_out_path,
                                  'save_video' : save_video,
                                  'detector' : detector,
                                  'checkpoint' : model_path,
                                  'cfg' : model_config_path,
                                  'pose_track' : True,
                                  'suffix' : suffix_str,
                                  'vis_fast' : vis_fast,
                                  'param' : [ idim_, nms_, conf_ ],
                                  'sp' : sp,
                                  'gpus' : gpus,
                                  'flip' : flip,
                                  'verbosity' : verbosity }
            inference.run(alphapose_argdict)

    elif program == 'demo_inference':

        if video_out_path: save_video_cmd = ['--visoutdir',video_out_path,'--save_video']
        else: save_video_cmd = ['','','']

        if vis_fast: visfast_cmd = '--vis_fast'
        else: visfast_cmd = ''

        def _run_AP(video_to_track_ffn, idim_, nms_, conf_, suffix_str):

            if suffix_str: suffix_cmd = ['--suffix',suffix_str]
            else: suffix_cmd = ['','']

            def _subprocess_AP(cmd):
                clear_line = True
                AP_out = subprocess.Popen( cmd, shell=True, capture_output=True,
                                           text=True)
                for l in iter(AP_out.stdout.readline,''):
                    if l:
                        if clear_line:
                            for _ in range(14): print('\b',end='',flush=True)
                            print()
                            clear_line = False
                        yield l
                AP_out.stdout.close()
                rc = AP_out.wait()
                if rc: raise subprocess.CalledProcessError(rc, cmd)
                if AP_out.stderr: print('\nstderr:\n',AP_out.stderr)

            AlphaPose_cmd = [ 'python3','scripts/demo_inference.py',
                              '--sp','--video',video_to_track_ffn,'--jsonoutdir',json_path,
                               save_video_cmd[0],save_video_cmd[1],save_video_cmd[2],
                              '--param', str(idim_), str(nms_), str(conf_) ,
                              '--detector',detector, '--verbosity', str(verbosity),
                              '--checkpoint',model_path,'--cfg',model_config_path,
                              '--pose_track',suffix_cmd[0],suffix_cmd[1],visfast_cmd ]

            if verbosity==2:
                if gpus == '-1':
                    for l in _subprocess_AP(AlphaPose_cmd): print(l)
                else:
                    try:
                        AP_out = subprocess.run( AlphaPose_cmd, check=True,
                                                 capture_output=True )
                        print("stdout :\n", AP_out.stdout)

                    except subprocess.CalledProcessError as e:
                        print('exit_code:', e.returncode)
                        print('stderror:\n',e.stderr)
            else:
                if gpus == '-1': subprocess.run( AlphaPose_cmd, shell=True )
                else: subprocess.run( AlphaPose_cmd, shell=False )

    else: raise Exception('value for argument "program" is invalid')

    def _one_posetrack( idim_, nms_, conf_ ):

        for fn in fnames:
            ffn = os.path.join(video_in_path,fn)
            split_fn = os.path.splitext(fn)
            fn_ne = split_fn[0] + trim_lbl
            if parlbl or log_path: par_str_short = f'[{idim_},{nms_},{conf_}]'
            if parlbl: suffix_str = f'_{par_str_short}{suffix}'
            else: suffix_str = suffix
            json_fn = f'AlphaPose_{fn_ne}{suffix_str}.json'
            new_file = True
            if skip_done: new_file = json_fn not in json_saved_fn
            if parlbl or log_path or verbosity:
                par_str_long = f'idim = {idim_}; nms = {nms_}; conf = {conf_}'
            if verbosity:
                print(f'{fn} :\n{par_str_long}')

            if (not skip_done) or new_file:

                if log_path: tracking_log_txt = [f'{fn}\n{par_str_long}\n']
                if verbosity or log_path: tic = time.time()

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
                    video_to_track_ffn = ''.join([ video_out_folder, '/', split_fn[0],
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
                _run_AP(video_to_track_ffn, idim_, nms_, conf_, suffix_str)

                if verbosity or log_path:
                    toc = round(time.time() - tic,3)
                    toc_str = f"computing time = {str(timedelta(seconds=toc))[:-3]} (H:M:S)\n"
                    if verbosity: print(toc_str,'\n')

                # Set audio to output video:
                if video_out_path and audio:
                    tracked_out_fn = ''.join([ 'AlphaPose_', split_fn[0], trim_lbl,
                                                suffix_str, split_fn[1] ])
                    tracked_video_ffn = os.path.join(video_out_path,tracked_out_fn)
                    setaudio( tracked_video_ffn, f'{audio_ffn}.{audio_ext}' )

                # save log:
                if log_path:
                    tracking_log_txt.append(toc_str)
                    txtlog_ffn = f'{log_path}/posetrack_log.txt'
                    tracking_log_txt.append('\n')
                    with open(txtlog_ffn, 'a') as output:
                        for t in tracking_log_txt:
                            output.write(t)

            elif verbosity: print('skipped')

    if not isinstance(idim,list): idim = [idim]
    if not isinstance(nms,list): nms = [nms]
    if not isinstance(conf,list): conf = [conf]

    for idim_ in idim:
        for nms_ in nms:
            for conf_ in conf:
                _one_posetrack( idim_, nms_, conf_ )
    os.chdir(cwd)

def poseprep( json_path, savepaths, vis={}, **kwargs ):
    '''
    Pre-process AlphaPose's tracking results.
    Args:
        json_path (str): Path of folder for input json AlphaPose tracking files.
        savepaths (dict):
                savepaths['parquet'] (str): Path to folder for pre-processed data parquet files.
                Optional:
                    savepaths['rawfig'] (str): Path to folder for raw data figures.
                    savepaths['prepfig'] (str): Path to folder for pre-processed data figures.
                    savepaths['log'] (str): Path to folder for pre-processing log file.
        Optional:
            vis (dict): Visualisation options.
                vis['show'] (bool): Show visualisation (independent of saving).
                vis['markersize'] (int,float): Marker size for raw data plots.
                vis['lwraw'] (int,float): Line width for raw data plots.
                vis['lwprep'] (int,float): Line width for pre-processed data plots.
            keypoints (list): Default =[0,1] ([x1,y1] for "Nose", assuming COCO format).
            kp_labels (list): Labels for keypoints. Default = ['x','y']
            n_indiv (int,str): Expected number of individuals to be tracked. Default = 'auto'
            sel_indiv (int,list,str): Selection of individuals with index in json file starting
                                      at 1. Default = 'all'
            skip_done (bool): Skip if corresponding preprocessed parquet file exists.
            suffix (str): Label to be added to the names of the resulting files.
            trange (list): Time-range selection [start,end]. Default = None
            scorefac (float): NMS score factor to discard raw data. 0 >= scorefac <= 1
                              Default = 0.7
            drdim (str,int,list): Dimensions to apply disjoint ranges for tracked individuals.
                                  'all' to try all dimensions. Works only if keypoint trajectories
                                  don't overlap. Default = None
            drlim_set (list): Set manual limits for disjoint ranges, only if json_path is a file or
                              is a folder with only one file. Format is nested lists for dimensions.
                              The order should be consistent with drdim. Default = None
                              Example: [[lim0_dim0,lim1_dim0], [lim0_dim1,lim1_dim1]]
            fillgaps (bool): Fill missing data with cubic spline. Default = True
            verbose (bool): Default = True
    Returns:
            drlim_file (list): Limits of disjoint ranges, only if json_path is a file and
                               drdim is not None.
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
    kp_labels = kwargs.get('kp_labels',['x','y'])
    if len(keypoints) > 2:
        raise Exception(''.join([ f'Currently only one point with two dimensions (x,y) \
                                    are allowed, but instead got this: {keypoints}' ]))
    n_indiv = kwargs.get('n_indiv','auto')
    sel_indiv = kwargs.get('sel_indiv','all')
    skip_done = kwargs.get('skip_done',True)
    suffix = kwargs.get('suffix',None)
    trange = kwargs.get('trange',None)
    scorefac = kwargs.get('scorefac',0.7)
    drdim = kwargs.get('drdim',None)
    drlim_set = kwargs.get('drlim_set',None)
    fillgaps = kwargs.get('fillgaps',True)
    verbose = kwargs.get('verbose',True)

    if vis['show'] or rawfig_path or prepfig_path:
        import matplotlib.pyplot as plt
        if plt.get_backend() == 'agg': # AlphaPose uses 'agg'
            from importlib import reload
            import matplotlib
            reload(matplotlib)
            reload(matplotlib.pyplot)

    if isinstance(sel_indiv,int): sel_indiv = [sel_indiv]

    if drdim is not None: from sklearn.cluster import HDBSCAN
    if isinstance(drdim,int): drdim = [drdim]
    elif isinstance(drdim,str): drdim = list(range(len(kp_labels)))

    parquet_fnames = utils.listfiles(savepaths['parquet'])

    json_fnames = utils.listfiles(json_path)
    json_path_is_file = os.path.isfile(json_path)
    if json_path_is_file: json_path = os.path.dirname(json_path)

    if drlim_set:
        assert drdim, 'argument "drdim" should have a value'
        assert len(drdim) == len(drlim_set), '"drdim" and "drlim_set" should be the same length'
        assert (len(json_fnames)==1) or json_path_is_file, 'argument "drlim_set" works only for one file'

    for json_fn in json_fnames:

        fn_ne = os.path.splitext(json_fn)[0]
        if verbose: print(f'{json_fn} :',end=' ')

        parquet_fn = f'{fn_ne}.parquet'
        new_file = True
        if skip_done: new_file = parquet_fn not in parquet_fnames

        if (skip_done is None) or new_file:

            if verbose: print('processing...')

            # Load data from JSON file produced by AlphaPose:
            data_raw_df = pd.read_json(json_path + '/' + json_fn)
            if n_indiv == 'auto': n_persons = data_raw_df.idx.max()
            else: n_persons = n_indiv

            # Reduce by removing unnecessary data:
            data_red_df = data_raw_df.drop(['category_id','keypoints','score','box'],axis=1)
            score_thresh = (data_raw_df.score.max() - data_raw_df.score.min()) * scorefac
            data_red_df = data_red_df[ data_raw_df.score >= score_thresh ]
            for lbl,i in zip(kp_labels,keypoints):
                data_red_df[lbl] = data_raw_df.keypoints.str[i]
            data_red_df.image_id = data_red_df.image_id.str.split('.').str[0].astype(int)
            if trange: t_loc = trange
            else: t_loc = [0,data_red_df.image_id.max()]
            idx_df_sel = (data_red_df.image_id >= t_loc[0]) & (data_red_df.image_id <  t_loc[1])
            data_red_df = data_red_df[idx_df_sel]
            if sel_indiv == 'all': persons_range = range(1,n_persons+1)
            else:
                n_persons = len(sel_indiv)
                persons_range = sel_indiv

            # Inspect and make plot of raw data:
            if vis['show'] or rawfig_path or log_path or drdim:
                if drdim:
                    if drlim_set:
                        drlim_file = drlim_set
                        i_drlim = 0
                    else: drlim_file = []
                if log_path: prep_log_txt = [fn_ne + '\n']

                n_series = len(keypoints)
                for i_s in range(n_series):
                    if vis['show'] or rawfig_path: plt.subplot(n_series,1,i_s+1)
                    n_frames = []
                    legend = []
                    for i_p in persons_range:
                        data_red_p_df = data_red_df[kp_labels[i_s]][data_red_df.idx == i_p]
                        if vis['show'] or rawfig_path:
                            data_red_p_df.plot(linewidth=vis['lwraw'],alpha=0.7)
                        n_frames.append(len(data_red_p_df))

                    if vis['show'] or rawfig_path or drdim:
                        this_series = data_red_df[kp_labels[i_s]]

                    if vis['show'] or rawfig_path:
                        this_series.plot( marker='.', linestyle='none',
                                          markersize=vis['markersize'], color='k')

                    if drdim and (i_s in drdim):
                        if drlim_set:
                            dim_max = this_series.max()
                            drlim_file[i_drlim] = [0] + drlim_file[i_drlim] + [dim_max]
                            drlim_series = drlim_file[i_drlim]
                            plot_hlines = True
                            i_drlim += 1
                        else:
                            mcs = int(len(this_series)/(n_persons*2))
                            clustering = HDBSCAN( min_cluster_size=mcs, store_centers="centroid",
                                                  metric="cityblock" )
                            clustering.fit(np.reshape(this_series, (-1, 1)))
                            centroids = np.squeeze(clustering.centroids_)
                            if centroids.size == n_persons:
                                drlim_series = centroids[:-1] + np.diff(centroids)/2
                                drlim_series = np.insert( drlim_series, [0,len(centroids)-1],
                                                          [0,this_series.max()] )
                                drlim_file.append(np.sort(drlim_series).tolist())
                                plot_hlines = True
                            else:
                                drlim_file.append(None)
                                print( ''.join([ 'Warning: no disjoint ranges for axis',
                                                 f'{i_s}', ' (', f'{kp_labels[i_s]}', ')' ]))
                                plot_hlines = False
                        if plot_hlines: plt.hlines( drlim_series[1:-1], 0, data_red_df.index.max(),
                                                    linestyles='dashed',
                                                    colors='tab:gray', linewidths=0.8 )

                    plt.ylabel(kp_labels[i_s])
                    if i_s == 0:
                        plt.legend( list(persons_range)+['all'],loc='upper right',
                                    bbox_to_anchor=(1.2, 1.02) )

                    if log_path:
                        mean_persons = sum(n_frames)/n_persons
                        for p in n_frames:
                            if p != mean_persons:
                                warning_frames = ''.join([ 'inconsistent frame count in '
                                                          f'{kp_labels[i_s]} {tuple(n_frames)}' ])
                                prep_log_txt.append( warning_frames+'\n' )
                                if verbose: print('Warning:',warning_frames)
                                break

                if rawfig_path or vis['show']:
                    plt.gcf().suptitle(f'{fn_ne}\nRaw (NMS score factor = {scorefac})')
                    plt.gcf().supxlabel('stacked frames (as in json file)')
                    plt.tight_layout()
                    if rawfig_path:
                        fig_ffn = rawfig_path + '/' + fn_ne + '_RAW.png'
                        plt.savefig(fig_ffn)
                    if vis['show']: plt.show()
                    else: plt.close(plt.gcf())

                if log_path or verbose:
                    if (data_red_df.idx.max()) != n_persons:
                        warning_idx = 'more idx than number of people in raw data'
                        if verbose: print('Warning:',warning_idx)
                        if log_path: prep_log_txt.append(warning_idx+'\n')

            # Apply disjoint ranges:
            if drdim:
                i_l = 0
                for i_drdim in drdim:
                    if isinstance(drlim_file[i_drdim],list):
                        for i_p, _ in enumerate(drlim_file[i_l][:-1]):
                            col_lbl = kp_labels[i_drdim]
                            idx_lo = data_red_df[col_lbl] > drlim_file[i_l][i_p]
                            idx_hi = data_red_df[col_lbl] <= drlim_file[i_l][i_p+1]
                            idx_sel = idx_lo & idx_hi
                            data_red_df.loc[idx_sel,'idx'] = i_p+1
                        drlim_file[i_l] = drlim_file[i_l][1:-1] # for output
                    i_l += 1

            # Rearrange such that each row is a frame (image_id):
            data_rar_df = pd.DataFrame( list(range(data_red_df.image_id.max() + 1)) ,
                                        columns=['image_id'] )
            for i_p in persons_range:
                red_df_cols = ['image_id'] + kp_labels
                red_df_idx = data_red_df.idx == i_p
                data_rar_df = data_rar_df.merge( data_red_df[red_df_cols][(red_df_idx)],
                                                 on='image_id', how='left',
                                                 suffixes=(f'_{i_p-1}',f'_{i_p}') )
            data_rar_df = data_rar_df.drop(['image_id'],axis=1)

            # Re-order and re-label columns in order from left to right as they appear in the image:
            # It is assumed that the persons don't relocate (e.g. they are sitting or standing in
            # one place). Indices are set to start at 0 to be consistent with Python indexing.
            new_order_x = [ x for x in data_rar_df.iloc[:,::2].median().sort_values().index]
            new_order_y = [ y.replace(kp_labels[0],kp_labels[1]) for y in new_order_x ]
            new_order_xy = []
            new_order_lbl = []
            i_c = 0
            for x,y in zip(new_order_x,new_order_y):
                new_order_xy.append(x)
                new_order_xy.append(y)
                new_order_lbl.append(f'{i_c}_{kp_labels[0]}')
                new_order_lbl.append(f'{i_c}_{kp_labels[1]}')
                i_c += 1
            data_rar_df = data_rar_df.reindex(new_order_xy, axis=1)
            data_rar_df.columns = new_order_lbl

            # Fill missing data:
            if fillgaps:
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
                txtlog_ffn = log_path + '/' + 'poseprep_log.txt'
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
                for i_s in range(n_series):
                    plt.subplot(n_series,1,i_s+1)
                    legend = []
                    names_cols = [ f'{n}_{kp_labels[i_s]}' for n in range(n_persons)]
                    for nc in names_cols:
                        data_rar_slice_df = data_rar_df[nc].iloc[ t_loc[0] : t_loc[1] ]
                        data_rar_slice_df.plot(linewidth=vis['lwprep'])

                        legend.append(nc.split('_')[0])
                    plt.ylabel(kp_labels[i_s])
                    if i_s == 0:
                        plt.legend(legend,loc='upper right', bbox_to_anchor=(1.2, 1.02))
                plt.suptitle(fn_ne+'\nPre-processed')
                plt.xlabel('time (video frames)')
                plt.tight_layout()
                if prepfig_path:
                    fig_ffn = prepfig_path + '/' + fn_ne + '_PREP.png'
                    plt.savefig(fig_ffn)
                if vis['show']: plt.show()
                else: plt.close(plt.gcf())

            # Write pre-processed data to a file:
            if savepaths['parquet']:
                parquet_ffn = preproc_path + '/' + parquet_fn
                data_rar_df.to_parquet(parquet_ffn)

            if verbose: print('done')

        else:
            if verbose: print('skipped')

    if (drdim is not None) and (len(json_fnames)==1): return drlim_file
