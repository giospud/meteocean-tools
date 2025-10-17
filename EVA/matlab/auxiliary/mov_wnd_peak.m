% dd    --> dates
% hh    --> sig. wave height
% tt    --> peak period
% th    --> peak direction
% delta --> window width

function [D,H,T,TH] = mov_wnd_peak(dd,hh,tt,th,delta)

n = numel(dd);
k = 1;
D = []; H = []; T = []; TH = [];

for i = 1:(n-delta)

    endi = i + delta;
     
    href  = hh(i:endi);    
    hmax = max(href);
    imax = i + find(href==hmax) - 1;

    % check on SWH peak
    it = round(i + (delta/2));

    % store variables if the peak falls in the middle of the window
    if it == imax(1) 

        D(k,1)  = datenum(dd(it));
        H(k,1)  = hh(it);
        T(k,1)  = tt(it);
        TH(k,1) = th(it);
     
        k = k+1;

    end

end

return


