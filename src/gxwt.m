function [xs, f, p1, p2] = gxwt(a1,a2,fps,minf,maxf,varargin)
% Wrapper for cwt.m, cwtensor.m, and genxwt.m
%
% Args:
%   a1, a2: multivariate signals, dim = [time, channel]
%   fps: sampling rate (Hz or fps)
%   minf: minimum frequency included (Hz)
%   maxf: maximum frequency included (Hz)
%   OPTIONAL (varargin):
%       type: 'all' (default): pseudo-variace for all channel pairs, or
%             'pairwise': pseudo-variance for corresponding channel pairs
%       Arguments passed to cwt.m (default = 'morse').
%
% Returns:
%   xs: cross-wavelet spectrum, dim = [frequency, time]
%   p1, p2: projection tensors, dim = [frequency, time, channel]
%
% Dependencies:
%   Wavelet Toolbox for Matlab
%   cwtensor.m, genxwt.m 
%
% Reference:
%   https://doi.org/10.1016/j.humov.2021.102894

inarr = {a1,a2}
type = "all"
types = {type,"pairwise"}
for i = 1:2 % loop economy :)
    s = size(inarr{i});
    if s(2) > s(1)
        inarr{i} = inarr{i}';
    end
    ii = find(contains(varargin,types{i}));
    if ii
        type = types{i};
        varargin(ii) = [];
    end
end
if isempty(varargin)
    varargin = {"morse"}
end
[w1, f] = cwtensor(inarr{1},fps,minf,maxf,varargin{:})
[w2, f] = cwtensor(inarr{2},fps,minf,maxf,varargin{:})
[xs, p1, p2] = genxwt(w1,w2,type)
