clearvars
close all
clc

addpath ./auxiliary/

indir  = './input/';
outdir = './output/'; 

Ith    = 24; % window's width 4 declustering [hr]

%% 1. load hindcast data
filein = 'Point_000230_out.dat';
[date,Hs,~,Tp,~,thp,~,~,~,~,~,~,Ny] = loadDICCA([indir filein]);

%% 2. select peaks through mov. windows
[D,H,~,TH] = mov_wnd_peak(date,Hs,Tp,thp,Ith);

%% 3. export peaks
fout  = [filein(7:12)  '_HsDp_DT_' num2str(Ith)];

fig=figure;
set(fig,'Position',[360 178 860 360])
dscatter(TH,H);hold on

xline([22.5:45:360],'k--')
xline([0:45:360],'k:')

xlabel('\theta_p [°N]')
ylabel('H_s [m]')
xticks([22.5:45:360])
xlim([0 360])

text(0.11,0.95,'NE','Units','Normalized');hold on
text(0.24,0.95,'EA','Units','Normalized');hold on
text(0.36,0.95,'SE','Units','Normalized');hold on
text(0.48,0.95,'SO','Units','Normalized');hold on
text(0.60,0.95,'SW','Units','Normalized');hold on
text(0.73,0.95,'WE','Units','Normalized');hold on
text(0.86,0.95,'NW','Units','Normalized');hold on

set(gca,'FontSize',12)

print([outdir fout '.jpg'],'-djpeg','-r250')