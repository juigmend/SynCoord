#!/usr/bin/env python
# coding: utf-8

'''Data visualisation.'''

import numpy as np
import matplotlib.pyplot as plt
from .utilities import frames_to_minsec_str, nd_iter

def xticks_minsec(n_frames,fps,interval_sec=20,start_sec=0):
    '''
    Convert xticks expressed in frames to format "minutes:seconds".
    '''
    interval_f = interval_sec*fps
    xticks_loc = list(range(round(start_sec),round(n_frames),round(interval_f)))
    if xticks_loc[-1] > (n_frames-interval_f/2): xticks_loc[-1] = n_frames
    else: xticks_loc.append(n_frames)
    xticks_lbl = []
    for f in xticks_loc: xticks_lbl.append( frames_to_minsec_str(f,fps) )
    return xticks_loc, xticks_lbl

def xticks_percent(x_percent,length_x):
    '''
    Make xticks as percentage.
    '''
    frac = 100/float(x_percent)
    length_section = round(length_x / frac)
    x_ticks = [t*length_section for t in range(round(frac)+1)]
    x_labels = [ round(t*x_percent) for t in range(round(frac)+1)]
    plt.xticks(x_ticks,x_labels)

def overlay_vlines(ax, loc, vlattr, numcolour='k', num_hvoffset=None):
    '''
    Args:
        ax: pyplot axis object where to overlay vertical lines.
        loc: list with the location of the lines, in horizontal axis units.
        vlatrr: string with one character for each of these:
                colour, style, width, f (full vertical) or b (bits at the top and bottom). 
                For example: 'r:2f' means red, dotted, width=2, full vertical line.
        Optional:
            numcolor: Colour for numbers. None for no numbers.
            num_hvoffset: horizontal and vertical offset for the numbers, as percentage of axes lengths.
    Returns:
        none
    '''
    if not vlattr: vlattr='r:2f'
    ylims = ax.get_ylim()
    if vlattr[3] == 'f':
        n_iters = 1
        ymin = [ylims[0]]; ymax = [ylims[1]]
    elif vlattr[3] == 'b':
        n_iters = 2
        the_bit = (abs(ylims[0]) + abs(ylims[1]))*0.1
        ymin = [ylims[0], ylims[1]-the_bit]
        ymax = [ylims[0]+the_bit, ylims[1]]
    else:
        raise Exception(f'Rightmost chracater in "vlattr" should be "f" or "b", but got "{vlattr[3]}".')
    for i in range(n_iters):
        ax.vlines( x = loc, ymin=ymin[i], ymax=ymax[i],
                   colors = vlattr[0],
                   linestyles = vlattr[1],
                   linewidths = int(vlattr[2]) )
    if numcolour:
        hv_offset = [0,0]
        n_sec = len(loc)
        if num_hvoffset:
            xlims = ax.get_xlim()
            h_unit = (xlims[1] - xlims[0])/n_sec
            hv_offset[0] = h_unit * num_hvoffset[0]
            v_unit = ylims[1] - ylims[0]
            hv_offset[1] = v_unit * num_hvoffset[1]
        for i_s in range(n_sec):
            ax.text( loc[i_s] + hv_offset[0],
                     ylims[0] + hv_offset[1],
                     i_s, rotation=0, color=numcolour,
                     horizontalalignment='center')

def visualise_ptdata( ptdata, **kwargs ):
    '''
    Visualise data of a PtData object. Normally the object should contain information to generate the
    visualisation, mostly in the 'vis' field. Default settings may be changed with optional arguments.
    Args:
        ptdata: PtData object, see help(PtData).
        Optional:
            vistype: 'line', 'spectrogram', or 'imshow'
            groupby: int, str, or list, indicating N-D array's dimensions to group.
                     'default' = use defaults: line = 0, spectrogram = -2
            vscale: float, vertical scaling.
            dlattr: string, data lines' attributes colour, style, and width (e.g. 'k-0.6')
            sections: display vertical lines for sections. True or False.
            vlattr: vertical lines' attributes, see help(overlay_vlines).
            snum_hvoff: list with horizontal and vertical offset factor for section numbers.
            y_label: label for vertical axis. 'default' uses ptdata.labels.main
            y_ticks: labels for vertical axis ticks, useful only when vistype = 'imshow'
            x_ticklabelling: labelling of horizontal axis;
                             'default´'
                             's' = 'time (seconds)',
                             '25%' = xticks as percentage
                             'dim x' = use ptdata.labels.dim[x], or None.
            figtitle: figure title. If None, ptdata.name.main will be used.
            axes: dimensions to visualise. One axis for 'line' and'spectrogram', two axes for 'imshow'.
            sel_list: selection to display with list, see help(select_data).
            savepath: full path (directories and filename with extension) to save as PNG
            **kwargs: selection to display with keywords, see help(select_data).
    '''
    kwargs = {**ptdata.vis,**kwargs}
    vistype = kwargs.pop('vistype','line')
    groupby = kwargs.pop('groupby','default')
    y_max = kwargs.pop('y_max',None)
    vscale = kwargs.pop('vscale',1)
    dlattr = kwargs.pop('dlattr',None)
    sections = kwargs.pop('sections',True)
    vlattr = kwargs.pop('vlattr','k:2f')
    snum_hvoff = kwargs.pop('snum_hvoff',[0,1.13])
    y_label = kwargs.pop('y_label','default')
    y_ticks = kwargs.pop('y_ticks',None)
    x_ticklabelling = kwargs.pop('x_ticklabelling','s')
    figtitle = kwargs.pop('figtitle',None)
    axes = kwargs.pop('axes',-1)
    sel_list = kwargs.pop('sel_list',None)
    savepath = kwargs.pop('savepath',None)

    if y_label == 'default': ylabel = ptdata.labels.main
    else: ylabel = y_label
    dlattr_ = [None,None,0.6]
    if dlattr is not None:
        dstr = ''
        for c in dlattr:
            if c.isalpha(): dlattr_[0] = c
            elif c.isdigit() or (c == '.'): dstr += c
            else: dlattr_[1] = c
        if dstr: dlattr_[2] = float(dstr)
    ptdata = select_data(ptdata,sel_list,**kwargs)
    data_dict = ptdata.data
    data_dict_keys = tuple(data_dict.keys())
    if not isinstance(axes,list): axes = [axes]
    lastaxis_lbl = ptdata.names.dim[axes[-1]]
# TO-DO: case no sections in annotations
    sections_lastaxis_exist = f'trimmed_sections_{lastaxis_lbl}s' in ptdata.topinfo.columns
    if (lastaxis_lbl in kwargs) and sections and sections_lastaxis_exist:
        print(f'Warning: sections are disabled for {lastaxis_lbl} selection.')
        sections = False
# TO-DO:
    # select time maybe by xlims instead of select_data.
    # sections could follow frame selection.
    if figtitle is None: figtitle = ptdata.names.main
    super_title = ''
    if vistype == 'line':
        if (groupby == 'default') and (data_dict[data_dict_keys[0]].ndim > len(axes)):
            groupby = [0]
    elif vistype in ('spectrogram','imshow'):
        if 'spectrogram' in vistype:
            super_title = 'Frequency Spectrum\n'
        ylabel = 'Hz'
    if groupby == 'default':
        if 'imshow' in vistype: groupby = -2
        else: groupby = None
    spt_y = 1
    if sections and sections_lastaxis_exist: spt_y = snum_hvoff[1]*1.1
    data_shape = list(data_dict[data_dict_keys[0]].shape)
    sing_dims = False
    i_1 = []
    if 1 in data_shape: # singleton dimensions
        i_1 = [i for i, x in enumerate(data_shape) if x == 1]
        if not isinstance(groupby,list): groupby = [groupby]
        groupby.extend(i_1)
        sing_dims = True
    if groupby is None: groupby = axes
    if not isinstance(groupby,list): groupby = [groupby]
    groupby = groupby + axes
    for i,v in enumerate(groupby):
        if v<0: groupby[i] = data_dict[data_dict_keys[0]].ndim+v
    groupby = list(dict.fromkeys(groupby)) # remove redundancy
    for i,gb in enumerate(groupby):
        if isinstance(gb,str):
            groupby[i] = ptdata.names.dim.index(gb)
    s = data_shape
    for ii in sorted(groupby, reverse=True):
        if s: del s[ii]
    if len(s) > 1: raise Exception(   'Maximum 2 dimensions allowed for display, ' \
                                    + 'therefore groupby should be for more dimensions.' )
    default_xtick_percentage_str = '25%'
    if x_ticklabelling is None:
        if 'isochronal' in figtitle:
            x_ticklabelling = default_xtick_percentage_str
    else:
        if ('%' not in x_ticklabelling) and ('isochronal' in figtitle):
            x_ticklabelling = default_xtick_percentage_str
    if x_ticklabelling is None:
        xlabel = ''
        x_ticklabelling = 'blank'
    elif x_ticklabelling == 's':
        xlabel = 'time (m:s)'
    elif 'dim' in x_ticklabelling:
        xlabel = ptdata.labels.dim[int(x_ticklabelling.split(' ')[1])]
    elif '%' in x_ticklabelling:
        xlabel = '%'
        xpercent = float(x_ticklabelling.replace('%',''))
    n_sel_top = len(data_dict)
    n_sp = int(np.prod(s)*n_sel_top)
    fig_height = n_sp*2.4*vscale
    i_sp = 1
    fig = plt.figure(figsize=(12,fig_height))
    if y_ticks:
        sp_yticks = []
        sp_axes = []
    for i_top in range(n_sel_top):
        fps = ptdata.topinfo['fps'].iloc[i_top]
        top_arr = data_dict[data_dict_keys[i_top]]
        new_i_top = True
        array_iterator = nd_iter( top_arr, lockdim=groupby )
        for vis_arr,i_ch,i_nd in array_iterator:
            if new_i_top: sp_title = '"'+ptdata.topinfo['Name'].iloc[i_top]+'"'
            else: sp_title = ''
            plt.subplot(n_sp,1,i_sp)
            if sing_dims: vis_arr = np.squeeze(vis_arr)
            if vistype in ('line','imshow'):
                len_lastdim = vis_arr.shape[-1]
                if 'line' in vistype:
                    plt.plot( vis_arr.T, color=dlattr_[0], linestyle=dlattr_[1],
                              linewidth=dlattr_[2] )
                elif 'imshow' in vistype:
                    if vis_arr.ndim != 2:
                        check_gb = abs(2-vis_arr.ndim)
                        raise Exception(''.join(['The number of dimensions for imshow is currently ',\
                                                 f'{vis_arr.ndim} but should be 2. To correct it ',\
                                                 'please check keyword argument groupby has ',
                                                 f'length = {check_gb}']))
                    plt.imshow(vis_arr,aspect='auto')
                    plt.gca().invert_yaxis()
                if x_ticklabelling == 's':
                    plt.xticks( *xticks_minsec(len_lastdim,fps) )
                elif xlabel == '%':
                    xticks_percent(xpercent,vis_arr.shape[axes[-1]])
                plt.xlim((0,len_lastdim))
            elif 'spectrogram' in vistype:
                _,_,t,_ = plt.specgram(vis_arr,Fs=fps,detrend='linear',scale='linear')
                if x_ticklabelling == 's':
                    plt.xticks( *xticks_minsec(t[-1],1,start_sec=t[0]) )
                elif xlabel == '%':
                    xticks_percent(xpercent,len(t))
            if x_ticklabelling == 'blank':
                plt.xticks([],[])
            if y_max: plt.ylim((None,y_max))
            if y_ticks and (vis_arr.ndim == 2):
                if isinstance(y_ticks,list): sp_yticks.append(y_ticks)
                elif isinstance(y_ticks,dict): sp_yticks.append(y_ticks[ data_dict_keys[i_top] ])
                sp_axes.append( plt.gca() )
            plt.ylabel(ylabel)
            if sections and sections_lastaxis_exist:
                overlay_vlines( plt.gca(), ptdata.topinfo['trimmed_sections_frames'].iloc[i_top],
                                vlattr, numcolour=[0.6,0.1,0.2], num_hvoffset=snum_hvoff )
            for i in i_ch:
                if isinstance(ptdata.labels.dimel[i],dict): # dict: different labels for each top array
                    sp_lbl = ptdata.labels.dimel[i][i_top][i_nd[i]]
                elif isinstance(ptdata.labels.dimel[i],list):
                    sp_lbl = ptdata.labels.dimel[i][i_nd[i]] # list: same labels for all top arrays
                else:
                    sp_lbl = ptdata.labels.dimel[i] # (singleton dim.) same labels for all top arrays
                sp_title = ''.join([sp_title,'\n',sp_lbl])
            plt.title(sp_title,y=spt_y)
            if new_i_top: new_i_top = False
            i_sp += 1
    fig.supxlabel(xlabel)
    plt.suptitle( super_title + figtitle , fontsize=16 )
    plt.tight_layout(rect=[0, 0.005, 1, 0.98])
    if y_ticks and (vis_arr.ndim == 2): # TO-DO: this leaves a bit too much space in between ticks
        for i_ax,spax in enumerate(sp_axes):
            yticks_loc = spax.get_yticks()
            if min(yticks_loc) < 0: yticks_loc = np.delete(yticks_loc,0)
            if max(yticks_loc) > (len(sp_yticks[i_ax])-1): yticks_loc = np.delete(yticks_loc,-1)
            yticks_lbl = [ str(round(sp_yticks[i_ax][int(i)],1)).rstrip('0').rstrip('.') \
                           for i in yticks_loc ]
            max_ytick = len(sp_yticks[i_ax])-1
            if max_ytick not in yticks_loc:
                yticks_loc = np.append(yticks_loc,max_ytick)
                yticks_lbl.append(str(round(sp_yticks[i_ax][-1],1)).rstrip('0').rstrip('.'))
                if (yticks_loc[-1] - yticks_loc[-2]) <= 1:
                    yticks_loc = np.delete(yticks_loc,-2)
                    del yticks_lbl[-2]
            sp_axes[i_ax].set_yticks(yticks_loc,labels=yticks_lbl)
    if savepath:
        plt.savefig(savepath + '.png')
