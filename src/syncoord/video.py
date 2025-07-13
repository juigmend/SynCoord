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
        path_in (str): Input video file or folder.
        path_out (str): Output video file or folder.
        Optional kwargs:
        (either trim or crop are optional)
            trim (list): [start,end] in seconds.
            crop (str): 'width:height:right:down' *
                Example: '400:200:50:10' results in 400 px. (pixels) of width, 200 px. of height,
                         at 50 px. right and 10 px. down from the top-left corner.
            skip_done (bool): Skip if output file exists. Default = False
            lbl (str): Suffix for the output file's name or '--auto'. Default = None
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
    if not crop: sp_cmd_b += ['-acodec', 'copy','-vcodec', 'copy']

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
        ID (str): Identification code at the end of the video's Youtube page URL.
                  Note: The Youtube ID follows this string in the URL: "www.youtube.com/watch?v="
        mode (str): 'download', 'preview', or 'info'. They may be combined.
        Optional kwargs:
            video_folder (str): Path to save the downloaded video file.
            prop_folder (str): path to save the properties file. Default is video_folder
            fn (str): Name for the resulting file. If None, ID will be used.
            maxh (int): Maximum height of video to download. Default = 720
            maxfps (int): Maximum frame rate of video to download. Default = 60
            ext (str): File extension (encapsulation format) for resulting video.
            verbose (bool)
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
        ffn_in (str): Filename of input video file, with full or relative path stem.
        ffn_ne_out (str): Filename of output audio file, with no extension, with full or relative
                          path stem. If None, the audio file will have the same name and will be
                          placed in the same folder, as the input ideo file.
    Returns:
        (str): Extension (format) of the audio file.
    Dependencies:
        ffmpeg and ffprobe installed in system (callable by command line).
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
        ffn_video_in (str): Filename of input video file.*
        ffn_audio_in (str): Filename of input audio file.*
        ffn_video_out (str): Filename of output audio file.* If None, the file will be saved in the
                             same folder as input video file, with same name plus label "+audio".
        * file names should include full or relative path
    Dependency:
        ffmpeg installed in system (callable by command line).
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
        video_in_path (str): Path for input video file or folder with input video files.
        json_path (str): Path of folder for resulting json tracking files.
        AlphaPose_path (str): Path of folder where AlphaPose code is.
        Optional kwargs:
            video_out_path (str): Path of folder for resulting tracking video files
                                  with superimposed skeletons. None = Don't make video.
            trim_range (list): [start,end] in seconds or 'end'.
            log_path (str): Path of folder for log file.
            skip_done (bool): Skip if corresponding json file exists in json_path, default = False
            idim (int,list[int]): Size of the detection network, multiple of 32.
            nms (float,list[float]): NMS threshold for detection in (0...1]
            conf (float,list[float]): Confidence threshold for detection in (0...1]
            parlbl (bool): Add [idim,nms,conf] to the names of the resulting files.
            suffix (str): Label to be added to the names of the resulting files.
            audio (bool): Extract audio from tracking video and add to AlphaPose video files.
                          Audio files will be saved in video_in_path, video files with added audio
                          will be saved in video_out_path.
            sp (bool): Run on a single process. Forcefully True if operating system is Windows.
            gpus (str): Index of CUDA device. Comma to use several, e.g. "0,1,2,3".
                        Use "-1" for cpu only. Default="0"
            program (str,None):
                     If str: 'inference' (module) or 'demo_inference' (script)
                              The former prints nicely on a Python IDE (e.g., Jupyter),
                              but may not work with GPU.
                     If None (default) and gpus = "-1": use module AlphaPose/inference.py
                     Else: run script AlphaPose/scripts/demo_inference.py
            flip (bool): Enable flip testing. It might improve accuracy.
            detector (str): See documentation for available detectors. Default = 'yolo'
            model (str): Path for pretrained model (A.K.A. checkpoint).
            config (str): Path for pretrained model's configuration file.
            vis_fast (bool): Simpler and faster visualisation. Default = True
            verbosity (int): 0 (minimal), 1 (progress bar), or 2 (full). Default = 1
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

    video_in_path = os.path.abspath(video_in_path)
    if os.path.isfile(video_in_path):
        video_in_folder = os.path.dirname(video_in_path)
        fnames = [os.path.basename(video_in_path)]
    else:
        video_in_folder = video_in_path
        fnames = utils.listfiles(video_in_path)

    json_path = os.path.abspath(json_path)
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
        video_out_folder = video_in_folder + '/trimmed'
        if not os.path.exists(video_out_folder): os.makedirs(video_out_folder)
    else:
        trim_lbl = ''
        video_out_folder = video_in_folder

    if skip_done:
        json_saved_fn = []
        for fn in os.listdir(json_path):
            ffn_json = os.path.join(json_path, fn)
            if os.path.isfile(ffn_json): json_saved_fn.append( fn )

    if audio and video_out_path:
        audio_out_folder = os.path.join(video_out_folder,'audio')
        if not os.path.exists(audio_out_folder): os.makedirs(audio_out_folder)

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
            ffn = os.path.join(video_in_folder,fn)
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
                vis['show'] (bool,str): True (default) = Show visualisation (independent of saving).
                                        'ind' to plot individuals separately.
                vis['markersize'] (float): Marker size for raw data plots. Default = 0.8
                vis['lwraw'] (float): Line width for raw data plots. Default = 4
                vis['lwprep'] (float): Line width for pre-processed data plots. Default = 2
            keypoints (int): Default = 0 (x1 and y1 of "Nose"). Currently only one point allowed.
                             See "Documentation on Keypoints" below.
            kp_labels (list): Labels for keypoints. Default = ['x','y']
            kp_horizontal (int): Keypoint index of horizontal axis. Default = 0
            n_indiv (int,str): Expected number of individuals to be tracked. Default = 'auto'
            sel_indiv (int,list[int],str): Selection of individuals with index in json file starting
                                           at 1. Default = 'all'
            skip_done (bool): Skip if corresponding preprocessed parquet file exists.
            suffix (str): Label to be added to the names of the resulting files.
            trange (list): Time-range selection in frames [start,end]. Default = None
            confac (float): Confidence score factor to discard raw data. 0 >= confac <= 1
                              Default = 0.5
            drdim (str,int,list[int]): Dimensions to apply classification of individuals. Clustering
                                       is used if drlim_set is not specified. 'all' to try all
                                       dimensions. Works only if keypoint trajectories in selected
                                       dimensions don't overlap.
            drlim_set (list[int]): Set manual limits of disjoint ranges to classify individuals,
                                   only if json_path is a file or a folder with only one file.
                                   Format is one list if one dimension or nested lists for more
                                   dimensions. The order should be consistent with drdim.
                                   Example: [[lim0_dim0,lim1_dim0], [lim0_dim1,lim1_dim1]]
            fillgaps (bool): Fill missing data with cubic spline. Default = True
            verbose (bool): Default = True
    Returns:
        drlim_file (list): Limits of disjoint ranges. Only if json_path is a file
                           or folder has one file, and drdim is not None. Otherwise empty list-
    Documentation on Keypoints depending on training dataset:
        COCO and MPII:
            Default: COCO (n=17), MPII (n=16)
            cmu/open: COCO (n=18), MPII (n=15)
            https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md
        Halpe:
            full (n=136), body (n=26), face (n=68), hands (n=42)
            https://github.com/Fang-Haoshu/Halpe-FullBody
    '''
    assert isinstance(savepaths,dict), 'Keyword argument "savepaths" should be dict.'
    assert savepaths['parquet'], 'savepaths["parquet"] not in input.'
    preproc_path = savepaths['parquet'] # TO-DO: save to numpy file
    rawfig_path = savepaths.get('rawfig',None)
    prepfig_path = savepaths.get('prepfig',None)
    log_path = savepaths.get('log',None)

    assert isinstance(vis,dict), 'Argument "vis" should be dict.'
    vis = {'show':'dim','markersize':0.8,'lwraw':4,'lwprep':2,**vis}
    keypoints = kwargs.get('keypoints',0)
    assert isinstance(keypoints,int), 'Currently only one point is allowed, and it has to be int.'
    kp_labels = kwargs.get('kp_labels',['x','y'])
    kp_horizontal = kwargs.get('kp_horizontal',0)
    n_indiv = kwargs.get('n_indiv','auto')
    sel_indiv = kwargs.get('sel_indiv','all')
    skip_done = kwargs.get('skip_done',True)
    suffix = kwargs.get('suffix','')
    trange = kwargs.get('trange',None)
    confac = kwargs.get('confac',0.5)
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
        if not any(isinstance(v, list) for v in drlim_set): drlim_set = [drlim_set]
        assert len(drdim) == len(drlim_set), '"drdim" and "drlim_set" should be the same length'
        assert (len(json_fnames)==1) or json_path_is_file, 'argument "drlim_set" works only for one file'

    cmap = plt.get_cmap("tab10")
    i_kp_x = keypoints*3
    idx_kpdim = [i_kp_x,i_kp_x+1]
    idx_kpdim_conf = idx_kpdim + [i_kp_x+2]
    drlim_file = []

    for json_fn in json_fnames:

        fn_ne = os.path.splitext(json_fn)[0]
        if verbose: print(f'{json_fn} :',end=' ')

        parquet_fn = f'{fn_ne}{suffix}.parquet'
        new_file = True
        if skip_done: new_file = parquet_fn not in parquet_fnames

        if (skip_done is None) or new_file:

            if verbose: print('processing...')

            # Load data from JSON file produced by AlphaPose:
            data_raw_df = pd.read_json(json_path + '/' + json_fn)
            if verbose: print('number of keypoints in file:',int(len(data_raw_df.keypoints[0])/3))
            if n_indiv == 'auto': n_persons = data_raw_df.idx.max()
            else: n_persons = n_indiv
            if sel_indiv == 'all': persons_range = range(1,n_persons+1)
            else:
                n_persons = len(sel_indiv)
                persons_range = sel_indiv
            idx_all_p = range(n_persons)

            # Reduce data:
            data_red_df = data_raw_df.drop(['category_id','keypoints','score','box'],axis=1)
            data_red_df.image_id = data_red_df.image_id.str.split('.').str[0].astype(int)
            for kplbl,i_kpdc in zip(kp_labels+['conf'],idx_kpdim_conf): # reconstruct with dimensions of selected keypoints
                data_red_df[kplbl] = data_raw_df.keypoints.str[i_kpdc]
            data_red_df = data_red_df.set_index('image_id')
            data_red_df.index.name = None
            index_max = data_red_df.index.max()
            if trange: t_loc = trange
            else: t_loc = [0,index_max]
            idx_df_sel = (data_red_df.index >= t_loc[0]) & (data_red_df.index <=  t_loc[1])
            data_red_df = data_red_df[idx_df_sel]
            n_frames_in = data_red_df.index.unique().size
            conf_min = data_red_df.conf.min()
            conf_thresh = ((data_red_df.conf.max() - conf_min) * confac) + conf_min
            data_red_df = data_red_df[ data_red_df.conf >= conf_thresh ]

            # Inspect, plot selected raw data, and cluster:
            if vis['show'] or rawfig_path or log_path or drdim:

                if drdim:
                    warning_n_clusters = False
                    if drlim_set:
                        drlim_file = drlim_set
                        i_drlim = 0
                    else:
                        mcs = int(n_frames_in/(n_persons*2))
                        clustering = HDBSCAN( min_cluster_size=mcs, store_centers="centroid",
                                              metric="cityblock" )
                if log_path: prep_log_txt = [fn_ne + '\n']
                n_kpdim = len(idx_kpdim)
                if vis['show'] == 'ind':
                    n_sp = n_kpdim*n_persons + n_kpdim*2 - 1
                    i_sp = 1
                x_lims = [data_red_df.index.min(), index_max]
                colours = []
                for i_kpdim in range(n_kpdim):
                    if vis['show'] == 'ind': new_kpdim = True
                    elif (vis['show'] is True) or rawfig_path: plt.subplot(n_kpdim,1,i_kpdim+1)
                    n_frames_p = []
                    colours.append([])
                    if drlim_set and (i_kpdim in drdim): drdim_means = []
                    kpdim_df = data_red_df[['idx',kp_labels[i_kpdim]]]
                    for i_n, i_p in enumerate(persons_range):
                        kpdim_p_df = kpdim_df[[kp_labels[i_kpdim]]][kpdim_df.idx == i_p]
                        if drlim_set: drdim_means.append(kpdim_p_df[kp_labels[i_kpdim]].mean())
                        if vis['show'] or rawfig_path:
                            if vis['show'] == 'ind':
                                if new_kpdim:
                                    i_sp += 1
                                    if i_kpdim > 0: i_sp += 1
                                plt.subplot(n_sp,1,i_sp)
                                i_sp += 1
                            kpdim_p_df[kp_labels[i_kpdim]].plot( linewidth=vis['lwraw'],
                                                                 alpha=0.7, color=cmap(i_n) )
                            plt.xlim(x_lims)
                            colours[i_kpdim].append(cmap(i_n))

                            if vis['show'] == 'ind':
                                kpdim_p_df[kp_labels[i_kpdim]].plot( marker='.', linestyle='none',
                                                                     markersize=vis['markersize'],
                                                                     color='k' )
                                plt.xticks(fontsize=7)
                                plt.yticks(fontsize=7)
                                if new_kpdim:
                                    plt.title(f'\n{kp_labels[i_kpdim]}')
                                    new_kpdim = False
                                plt.legend([i_p], loc='upper right', bbox_to_anchor=(1.2, 1.02))
                        n_frames_p.append(len(kpdim_p_df))

                    if vis['show'] or rawfig_path or drdim:
                        this_dim_all = kpdim_df[kp_labels[i_kpdim]]
                    if vis['show'] == 'dim':
                        this_dim_all.plot( marker='.', linestyle='none',
                                           markersize=vis['markersize'], color='k' )

                    #  Classify individuals with disjoint ranges:
                    if drdim and (i_kpdim in drdim):
                        if drlim_set:
                            drlim_dim = drlim_file[i_drlim]
                            plot_hlines = True
                            i_drlim += 1
                            idx_all_p = np.argsort(drdim_means)
                        else: # estimate by clustering
                            clustering.fit(np.reshape(this_dim_all, (-1, 1)))
                            centroids = np.squeeze(clustering.centroids_)
                            drlim_dim = centroids[:-1] + np.diff(centroids)/2
                            drlim_dim = np.insert( drlim_dim, [0,len(centroids)-1],
                                                      [0,this_dim_all.max()] )
                            drlim_file.append(np.sort(drlim_dim).tolist())
                            if centroids.size != n_persons:
                                warning_n_clusters = True
                                print( ''.join([ 'Warning: Automatic classification of individuals',
                                                f' not applied with axis {i_kpdim} '
                                                f'({kp_labels[i_kpdim]})\n         because the number',
                                                f' of found clusters is {centroids.size},\n        ',
                                                f' but the expected number of individuals is',
                                                f' {n_indiv}.']))
                        if vis['show'] == 'dim':
                            plt.hlines( drlim_dim[1:-1], 0, data_red_df.index.max(),
                                        linestyles='dashed', colors='tab:gray', linewidths=0.8 )

                    if drdim and (i_kpdim==(n_kpdim-1)) and warning_n_clusters:
                        print('This might be solved by increasing the value for argument "confac"')

                    if vis['show'] == 'dim':
                        plt.ylabel(kp_labels[i_kpdim])
                        if i_kpdim == 0:
                            plt.legend( list(persons_range)+['all'],loc='upper right',
                                        bbox_to_anchor=(1.2, 1.02) )

                    if log_path:
                        mean_persons = sum(n_frames_p)/n_persons
                        for p in n_frames_p:
                            if p != mean_persons:
                                warning_frames = ''.join([ 'inconsistent frame count in '
                                                          f'{kp_labels[i_kpdim]} {tuple(n_frames_p)}' ])
                                prep_log_txt.append( warning_frames+'\n' )
                                if verbose: print('Warning:',warning_frames)
                                break

                if rawfig_path or vis['show']:
                    plt.gcf().suptitle(f'{fn_ne}\nRaw (confidence factor = {confac})')
                    plt.gcf().supxlabel('time (stacked video frames)')
                    if vis['show'] == 'dim': plt.tight_layout()
                    if rawfig_path:
                        fig_ffn = rawfig_path + '/' + fn_ne + suffix + '_RAW.png'
                        plt.savefig(fig_ffn,bbox_inches='tight')
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
                data_red_df.idx = 0
                for i_drdim in drdim:
                    if (len(drlim_file[i_drdim]) -1) == n_persons:
                        for i_p, _ in enumerate(drlim_file[i_l][:-1]):
                            col_lbl = kp_labels[i_drdim]
                            idx_lo = data_red_df[col_lbl] > drlim_file[i_l][i_p]
                            idx_hi = data_red_df[col_lbl] <= drlim_file[i_l][i_p+1]
                            idx_sel = idx_lo & idx_hi
                            data_red_df.loc[idx_sel,'idx'] = i_p+1
                            if sum(idx_sel) > n_frames_in:
                                data_sel_df = data_red_df.loc[idx_sel]
                                data_sel_df = data_sel_df.sort_values('conf',ascending=False)
                                data_sel_df = data_sel_df[~data_sel_df.index.duplicated(keep='first')]
                                data_red_df = data_red_df.loc[~idx_sel]
                                data_red_df = pd.concat([data_red_df,data_sel_df])
                    else:
                        if verbose: print( ''.join [ "Warning: disjoint ranges for dimension ",
                                                    f"{i_drdim} not applied as number of ranges ",
                                                     "doesn't mach number of individuals ",
                                                    f"({n_persons})" ])
                    i_l += 1
                data_red_df = data_red_df[data_red_df.idx>0]
                ata_red_df = data_red_df.sort_index()

            # Rearrange such that each row is a frame:
            data_rar_df = pd.DataFrame( index=range(n_frames_in) )
            for i_p in persons_range:
                data_rar_df = data_rar_df.join( data_red_df[kp_labels][data_red_df.idx == i_p],
                                                lsuffix=f'_{i_p-1}', rsuffix=f'_{i_p}' )

            colnames = list(data_rar_df.columns)
            new_last_colnames = [f'{s}_{i_p}' for s in colnames[-n_kpdim:]]
            colnames[-n_kpdim:] = new_last_colnames
            data_rar_df.columns = colnames

            # Re-order and re-label columns in order from left to right as they appear in the image,
            # assuming that the individuals don't relocate (e.g. they are sitting or standing in
            # one place). Indices are set to start at 0 to be consistent with Python indexing.
            idx_h = []
            for i_c, cn in enumerate(data_rar_df.columns):
                if kp_labels[kp_horizontal] == cn[0]:
                    idx_h.append(i_c)
            sorted_df = data_rar_df.iloc[:,idx_h].median().reset_index().sort_values(0)
            idx_new_order = list(sorted_df.index)
            new_order_h = list(sorted_df['index'])
            new_order_lists = []
            colours_ra = []
            for i_kpdim in range(n_kpdim):
                colours_ra.append([])
                for i_no in idx_new_order: colours_ra[i_kpdim].append(colours[i_kpdim][i_no])
                if i_kpdim == kp_horizontal:
                    new_order_lists.append(new_order_h)
                else:
                    new_order_dim = [ d.replace(str(kp_labels[kp_horizontal]),kp_labels[i_kpdim])
                                      for d in new_order_h ]
                    new_order_lists.append(new_order_dim)
            new_order_all = []
            new_order_lbl = []
            i_np = 0
            for i_p,_ in enumerate(persons_range):
                for i_kpdim in range(n_kpdim):
                    new_order_all.append(new_order_lists[i_kpdim][i_p])
                    new_order_lbl.append(f'{i_np}_{kp_labels[i_kpdim]}')
                i_np += 1

            data_rar_df = data_rar_df.reindex(new_order_all, axis=1)
            data_rar_df.columns = new_order_lbl

            # Fill missing data:
            if fillgaps:
                found_nan = data_rar_df.isnull().values.any()
                if found_nan:
                    data_rar_df = data_rar_df.interpolate(limit_direction='both',method='cubicspline')
                    if log_path or verbose:
                        warning_interp = 'missing raw data have been interpolated'
                        if verbose: print('Warning:',warning_interp)
                        if log_path: prep_log_txt.append(warning_interp+'\n')

            # save log:
            if log_path:
                txtlog_ffn = log_path + '/' + 'poseprep_log.txt'
                prep_log_txt.append('\n')
                with open(txtlog_ffn, 'a') as output:
                    for t in prep_log_txt:
                        output.write(t)

            # Plot pre-processed data:
            i_sp = 1
            for i_s in range(n_kpdim):
                if vis['show'] == 'ind': new_series = True
                else:
                    plt.subplot(n_kpdim,1,i_s+1)
                    legend = []
                names_cols = [ f'{n}_{kp_labels[i_s]}' for n in range(n_persons)]
                for i_nc, nc in zip(idx_all_p,names_cols):
                    if vis['show'] == 'ind':
                        if new_series:
                            i_sp += 1
                            if i_s > 0: i_sp += 1
                        plt.subplot(n_sp,1,i_sp)
                        i_sp += 1

                    data_rar_slice_df = data_rar_df[nc]
                    this_colour = colours_ra[i_s][i_nc]
                    data_rar_slice_df.plot(linewidth=vis['lwprep'],color=this_colour)

                    nc_num = nc.split('_')[0]
                    if vis['show'] == 'ind':
                        plt.xlim(x_lims)
                        plt.xticks(fontsize=7)
                        plt.yticks(fontsize=7)
                        if new_series:
                            plt.title(f'\n{kp_labels[i_s]}')
                            new_series = False
                        plt.legend([nc_num], loc='upper right', bbox_to_anchor=(1.2, 1.02))
                    else: legend.append(nc_num)

                if vis['show'] == 'dim':
                    plt.ylabel(kp_labels[i_s])
                    if i_s == 0:
                        plt.legend(legend,loc='upper right', bbox_to_anchor=(1.2, 1.02))
            plt.suptitle(fn_ne+'\nPre-processed')
            plt.xlabel('time (video frames)')
            if vis['show'] == 'dim': plt.tight_layout()
            if prepfig_path:
                fig_ffn = prepfig_path + '/' + fn_ne + suffix + '_PREP.png'
                plt.savefig(fig_ffn,bbox_inches='tight')
            if vis['show']: plt.show()
            else: plt.close(plt.gcf())

            # Write pre-processed data to a file:
            if savepaths['parquet']:
                parquet_ffn = preproc_path + '/' + parquet_fn
                data_rar_df.to_parquet(parquet_ffn)

            if verbose and not vis['show']: print('done')
        else:
            if verbose: print('skipped')

    return drlim_file
