% this function extracts extremes through the moving window approach
% given as inputs:
% - times (datetime format)
% - water levels zR
% - window's width dT (expressed in days)
% - threshold level zth (expressed in m according to local datum)

function [peak_time,peak_z,lambda] = select_mw(time,zR,dT,zth)

[zpk,locs] = findpeaks(zR);
tpk        = time(locs);

minDist = days(dT); % min distance between peaks

% initialize local peaks
te   = tpk(1);
ze   = zpk(1);

for i = 2:length(tpk)

    if tpk(i) - te(end) < minDist
        
        if zpk(i) > ze(end) % keep the highest
            te(end) = tpk(i);
            ze(end) = zpk(i);
        end

    else % keep both
        te(end+1) = tpk(i);
        ze(end+1) = zpk(i);
    end

end

peak_time = te(ze>zth); % time of the peak
peak_z    = ze(ze>zth); % magnitude of the peak

unique_years = unique(year(peak_time));
Ny           = length(unique_years);
lambda       = length(peak_z)/Ny;

return