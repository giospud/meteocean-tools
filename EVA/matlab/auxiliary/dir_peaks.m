function [maxH,maxTH] = dir_peaks(H,TH,dtheta) 

edges = 0:dtheta:360;
[~,~,bin] = histcounts(TH, edges);

nbins = length(edges)-1;

% position of the max Hs within each populated bin
valid  = bin > 0;
idxMax = accumarray(bin(valid), (1:numel(TH))', [nbins 1], ...
    @(ix) ix(find(H(ix)==max(H(ix)),1,'first')), NaN);

% extract the maxima
maxH  = NaN(nbins,1);
maxTH = NaN(nbins,1);

validIdx = ~isnan(idxMax);
maxH(validIdx)  = H(idxMax(validIdx));   % SWH
maxTH(validIdx) = TH(idxMax(validIdx));  % PEAK DIRECTION

% select relative maxima --> relevant peaks
[~,iloc] = findpeaks(maxH);
maxH  = maxH(iloc);
maxTH = maxTH(iloc);
% npks  = length(maxH);

% % loop through the peaks
% fetch_width = 45;
% ndata       = nan(npks,1);
% 
% for i = 1:npks
% 
%     lt = maxTH(i) - fetch_width/2;
%     ut = maxTH(i) + fetch_width/2;
% 
%     xpoly = [lt maxTH(i) ut];
%     ypoly = [0 maxH(i) 0];
% 
%     ndata(i,1) = numel(find(inpolygon(TH,H,xpoly,ypoly) > 0));

return
