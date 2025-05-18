'''
Run syncoord.video.posetrack on SLURM.
Edit lines in the script indicated with an arrow like this: <---
'''

# Declare paths:

syncoord_path = "/projappl/project_141/SynCoord/src"     # <--- syncoord code
AlphaPose_path = "/projappl/project_141/AlphaPose"       # <--- AlphaPose code

data_folder_path = "/scratch/project_141/data"           # <--- project's data folder

video_in_path = data_folder_path+"/input_video"          # <--- input video files

# Paths for resulting files:
pose_tracking_path = data_folder_path + '/pose_tracking' # <---
video_out_path = pose_tracking_path+'/video'             # <--- video with superimposed skeletons
json_path = pose_tracking_path+'/tracking'               # <--- tracking (json format)

#-------------------------------------------------------------------------------

# Pose Detection and Tracking Parameters:

ptrack_kwargs = {'idim': 608,             # <--- input dimensions (proportional to resources)
                 'nms': 0.6,              # <--- NMS threshold (proportional to resources)
                 'conf': 0.1,             # <--- confidence threshold (inversely proportional to resources)
                 'trim_range': [0,5],     # <--- video range (seconds or 'end')
                 'parlbl': True,          # <--- parameters label in file names
                 'suffix': '_gpu',        # <--- add string to file name
                 'verbosity': 0,          # <--- verbosity level: 0 (minimal), 1 (pbar) , 2 (full)
                 'skip_done': False,      # <--- skip if resulting json file exists
                 'gpus': '0'              # <--- (str) index of gpu to use, separated by comma if more than one
                }
################################################################################

import sys

sys.path.append(syncoord_path)
from syncoord import video as sc_video

ptrack_kwargs['video_out_path'] = video_out_path
ptrack_kwargs['log_path'] = pose_tracking_path
ptrack_kwargs['sp'] = True

# detectors 'yolo' (YOLOv3, default) and 'tracker' seem not to work in Linux
ptrack_kwargs['detector'] = 'yolox'
ptrack_kwargs['model'] = AlphaPose_path + '/pretrained_models/halpe26_fast_res50_256x192.pth'
ptrack_kwargs['config'] = AlphaPose_path + '/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'

sc_video.posetrack( video_in_path, json_path, AlphaPose_path, **ptrack_kwargs )
