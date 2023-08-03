clear all
close all

%% Costant Input
g=9.81;
rhow=1026;
rhos=2650;
ni=0.000001;
s=rhos/rhow;
d=0.0005;
d=0.01
%DD=(g*(s-1)/(ni^2))^(1/3)*d;  % Dimensionless particle size parameter

%% Dean A constat evaluation
ws1=[0.001:0.001:1];  % ws in cm/s
ws2=ws1/10;             % ws in m/s

A1=0.067*ws1.^0.44;
A2=0.5*ws2.^0.44;

plot(ws1,A1)
hold on
plot(ws1,A2)

%% Fall velocity calculation
A=0.5*ws(d,1)^0.44;



