clear all 
close all

g=9.81;
K=0.4;
rhos=2650;
rhow=1026;
s=rhos/rhow;
P=0.4;
kk=0.78;

waves=load('Point_000296_ext.dat');
dates=datetime(waves(:,1),waves(:,2),waves(:,3),waves(:,4),0,0);
t_inizio=datetime(dates(1));
t_fine=datetime(dates(end));
Hs=waves(:,5);
Tm=waves(:,6);
Tp=waves(:,7);
Lm=waves(:,11);
Dirm=waves(:,8);
Dirp=waves(:,9);
sprd=waves(:,10);

cstDeg=[100];        % Coastline
th_i=cstDeg+20;     
th_f=cstDeg+180-20;
I=(Dirp<th_i | Dirp>th_f);
Hso=Hs;
Hs(I)=0; % Filtering Hs only form open water
%plot(Hso)
%hold on
%plot(Hs)

th=Dirp-(cstDeg+90);
th(I)=0;
Qs=K*g^0.6*(Hs.^2.4).*(Tp.^0.2).*(cosd(th).^1.2).*sind(th)./ ...
    (8*(s-1)*(1-P)*2^1.4*pi^0.2*kk^0.4);

%polarplot(alpha,Qpos,'-')
%hold on
%polarplot(alpha,-Qneg,'--')
polarplot(Dirp,Qs)
ax=gca;
d=ax.ThetaDir;
ax.ThetaZeroLocation='top';
ax.ThetaDir='clockwise';
thetatickformat('degrees');
%thetaticks([0 15 30 45 60 75 90 105 120 135 150 165 180])

