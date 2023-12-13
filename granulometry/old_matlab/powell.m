clear all
close all

H=3;
T=9;
L=9.81*T^2/(2*pi);
d50=0.02;

pr=(6.38+3.25*log(H/L))*H;
pc=-0.23*(H*T*sqrt(9.81)/(d50^1.5))^(-0.588)*H*L/d50;
hc=(2.86-62.69*H/L+443.29*(H/L)^2)*H;
if H/L<0.03
    pt=1.73*(H*T*sqrt(9.81)/(d50^1.5))^(-0.81)*H*L/d50;
else
    pt=(55.26+41.24*(H^2/(L*d50))+4.90*(H^2/(L*d50))^2)*d50;
end
if H/L<0.03
    ht=(-1.12+0.65*(H^2/(L*d50))-0.11*(H^2/(L+d50)^2))*H;
else
    ht=(-10.41-0.025*(H^2/(d50^1.5*sqrt(L)))-7.5*10^(-5)*(H^2/(d50^1.5*sqrt(L)))^2)*d50;
end
pb=28.77*(H/d50)^0.92*d50;
hb=-0.87+(H/L)^0.64*L;

xx=[];
yy=[];
% L0
xx=[[pr,pc] xx];
yy=[[0.8*hc,hc] yy];
% L1
xdv=linspace(pc,0,21)';
if H/L<0.03
    n1=0.84+23.93*(H/L);
else
    n1=1.56;
end
ydv=hc*(xdv./pc).^n1;
xx=[xx xdv'];
yy=[yy ydv'];
% L2
xdv=linspace(0,pt,21)';
n2=0.84-16.49*(H/L)+290.16*(H/L)^2;
ydv=-ht*(xdv./pt).^n2;
xx=[xx xdv'];
yy=[yy ydv'];
% L3
xdv=linspace(pt,pb,21)';
n3=1.005;
ydv=-ht-(hb-ht)*((xdv-pt)./(pb-pt)).^n3;
xx=[xx xdv'];
yy=[yy ydv'];

plot(xx,yy,'color',[0.9290 0.6940 0.1250])
hold on
plot([0,pb],[0,0],'color',[0 0.4470 0.7410])
axis equal

