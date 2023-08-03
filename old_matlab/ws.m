function vv = ws(dd,n)
% Function for Fall Velocity ws
% ws(d50,n) 
% - d50 is the median diameter
% - n set the formula
%
% Available Formula:
% n=1 - Hallrmeier(1981)
% n=2 - Soulsby (1997)
% n=3 - Dietrich (1982)
% n=4 - Van Rijn (1984)
%

g=9.81;
rhow=1026;
rhos=2650;
ni=0.000001;
s=rhos/rhow;
Ds=(g*(s-1)/(ni^2))^(1/3)*dd;  % Dimensionless particle size parameter
Rp=sqrt(g*(s-1)*dd^3)/ni;

switch(n)
    case 1 % Hallermeir (1981)
        if Ds^3<=39
            vv=ni*Ds^3/(18*dd);
        end
        if (Ds^3>39) && (Ds^3<10000)
            vv=ni*Ds^2.1/(6*dd);
        end
        if  Ds^3>=10000
            vv=1.05*ni*Ds^1.5/dd;
        end
    case 2 % Soulsby (1997)
        vv=ni/dd*((10.36^2+1.049*Ds^3)^0.5-10.36);
    case 3 % Dietrich (1982)
        vv=sqrt((s-1)*g*dd)*exp(-2.891394  ...
            + 0.95296*log(Rp) - 0.056835*(log(Rp))^2 ... 
            - 0.002892*(log(Rp))^3 + 0.000245*(log(Rp))^4);
    case 4 % Van Rijin (1984)
end

end