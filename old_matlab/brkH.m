function hh = brkH(h,T,ss,n)
% Function to evaluate breaking height
% Hb=brkH(h,T,ss,n)
% Input:
% - h: water depth
% - T: wave period
% - ss: bed slope
% - n: formula to be used
%
% n=1 -> Goda for shallow waters
% n=2 -> proportional to depth (gamma*h)

g=9.81;
L0=g*T^2/(2*pi);
A=0.18;
switch n
    case 1 % Shallow water with Goda
        hh=A*L0*(1-exp(-1.5*pi*h/L0*(1+15*ss^(4/3))));
    case 2 % Negligible depths with proportionality
        hh=0.727*h;
end