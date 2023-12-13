clear all
close all
clc
%%
%caricamento dati  come colonna execl
d=[19.000 16.000 12.500 11.200 9.500 8.000 6.300 4.750 3.350 2.800 2.000 0.850 0.075];%maglia setacci
q=[0.0 49.2 13.4 10.2 10.4 91.6 379.8 315.4 78.6 1.6 1.0 0.6 0.2]; %peso trattenuto
passato_perc=[100.0 94.8 93.4 92.4 91.3 81.6 41.7 8.6 0.4 0.2 0.1 0.0 0.0];
fi=-log2(d);

% curva granulometrica
q_tot=sum(q); %ot
n=length(d);
q_kept=zeros;
for i=2:n

    q_kept(i)=q(i)+q_kept(i-1);  %tot trattenuto 

end
p=(q_tot-q_kept)*100/q_tot; %percentuale passata

figure(1)
semilogx(d,p)
grid on
ylabel('% by weight passing the sieve [%]')
xlabel('Grain Size [mm]')
title('Granulometric curve')

%interpolazione per ricavare i diametri
d50=((50-p(7))/(p(6)-p(7)))*(d(6)-d(7))+d(7); 
fi50=-log2(d50);
d_50=d50/1000; %diametro in metri


%%
%seconda parte
%valluto Lmo con hp di essere in deep water 
g=9.81;
Hs=3.2; %m
T=8.8 %s
Lom=g*(T^2)/(2*pi); %120 m
rapporto=Hs/Lom; %viene minore di 0.03 quindi uso  questo come discriminante 

%formule 1°slide
pr=Hs*(6.38+3.25*log(Hs/Lom));
pc=(Hs*Lom/d_50)*((-0.23)*(Hs*T*g^0.5*d_50^1.5)^(-0.588));  %ERRORE
hc=Hs*(2.86-62.69*(Hs/Lom)+443.29*(Hs/Lom)^2);
pt=Hs*Lom/d_50*(1.73*(Hs*T*g^0.5*d_50^1.5)^(-0.81));        %ERRORE
pb=d_50*(28.77*(Hs/d_50)^0.92);
hb=Lom*(Hs/Lom)^0.64*(-0.87);
ht=Hs*(-1.12+0.65*(Hs^2/(Lom*d_50))-0.11*((Hs^2)/(Lom*d_50))^2);

%stesse formule ma con diamtero in mm
pc_mm=Hs*Lom/d50*(-0.23)*(Hs*T*g^0.5*d50^1.5)^(-0.588);
pt_mm=Hs*Lom/d50*(1.73*(Hs*T*g^0.5*d50^1.5)^(-0.81));
pb_mm=d50*(28.77*(Hs/d50)^0.92);
ht_mm=Hs*(-1.12+0.65*(Hs^2/(Lom*d50))-0.11*((Hs^2)/(Lom*d50))^2);

%formule 3°slide
n1=0.84+23.93*(Hs/Lom);
n2=0.84-16.49*Hs/Lom+290.16*(Hs/Lom)^2;
n3=0.45;
%%
%%PROFILO POWELL
%plotto il modello di dean con le grandezze in m
figure 
x1=pc:0.01:0; %ho generato il primo vettore
y1=hc*(x1./pc).^n1; %primo tratto profilo
plot(x1,y1)
x2=0:0.01:pt; %ho generato il secondo vettore per il secondo tratto
y2=ht*(x2./pt).^n2; %secondo tratto profilo
hold on
plot(x2,y2)
x3=pt:0.01:pb;%ho generato il terzo vettore per il terzo tratto
y3=ht+(hb-ht).*((x3-pt)/(pb-pt)).^n3; %terzo tratto profilo
hold on
plot(x3,y3)
xlabel('Distance to the shore [m]')
ylabel('Depth [m]')
title('Powell profile')


%plotto il modello di dean con le grandezze in mm (NON MI TORNANO
%DIMENSIONI)
figure 
x1=pc_mm:0.01:0; %ho generato il primo vettore
y1=hc.*(x1./pc_mm).^n1; %primo tratto profilo
plot(x1,y1)
x2=0:0.01:pt_mm; %ho generato il secondo vettore per il secondo tratto
y2=ht.*(x2./pt_mm).^n2; %secondo tratto profilo
hold on
plot(x2,y2)
x3=pt_mm:0.01:pb_mm;%ho generato il terzo vettore per il terzo tratto
y3=ht_mm+(hb-ht_mm)*((x3-pt_mm)/(pb_mm-pt_mm)).^n3%terzo tratto profilo
hold on
plot(x3,y3)
xlabel('Distance to the shore [m]')
ylabel('Depth [m]')
title('Powell profile')



