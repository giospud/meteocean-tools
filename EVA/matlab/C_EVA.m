clearvars
close all
clc

addpath ./auxiliary/

indir  = './input/';
outdir = './output/';

Ith    = 24;            % window's width [hr]
hth    = 2;             % SWH threshold
thlim  = [180 247.5];   % directional fetch
nboot  = 1000;          % number of bootstrap for CI computation
alpha  = 0.05;          % confidence level (1-alpha)

%% 1. load data
filein = 'Point_000230_out.dat';
[date,Hs,~,Tp,~,thp,~,~,~,~,~,~,Ny] = loadDICCA([indir filein]);

%% 2. select peaks through mov. windows and for selected fetch
[D,H,T,TH] = mov_wnd_peak(date,Hs,Tp,thp,Ith);

H   = H(TH>=thlim(1) & TH<thlim(2));
T   = T(TH>=thlim(1) & TH<thlim(2));
Hgp = H(H>hth);
Tgp = T(H>hth);

%% CALCOLO PROB. NON ECCEDENZA
nExc   = length(Hgp);
lambda = nExc/Ny;

% empirical Tr
ecdf = [1:nExc]./(nExc + 1);
eTr  = 1./(lambda.*(1-ecdf));

parmhat = gpfit(Hgp-hth);

sh = parmhat(1);    % shape (epsilon)
sc = parmhat(2);    % scale (sigma)

Tr = 1:1000;              % return periods
P  = 1 - 1./(lambda*Tr);  % prob. of non exceedance

He = gpinv(P,sh,sc,hth);

% non-parametric bootstrap 4 CI computation
He_boot = nan(nboot,length(Tr));
for i = 1:nboot   
    x = randsample(Hgp,length(Hgp),'true');
    try
        parmhat_b = gpfit(x - hth);
        sh_b = parmhat_b(1); sc_b = parmhat_b(2);
        He_boot(i, :) = gpinv(P, sh_b, sc_b, hth);
    catch
        He_boot(i, :) = NaN; % se fallisce la stima
    end
end

He_low  = quantile(He_boot, alpha/2, 1);
He_high = quantile(He_boot, 1 - alpha/2, 1);

fig=figure;
set(fig,'Position',[360 180 760 360])
semilogx(eTr,sort(Hgp),'o','Color',[.7 .7 .7],'MarkerFaceColor','w');hold on
semilogx(Tr,He,'-k');hold on
semilogx(Tr,He_low,'--k');hold on
semilogx(Tr,He_high,'--k');
xlabel('T_R [y]')
ylabel('H_s [m]')
set(gca,'FontSize',12)

fout  = [filein(7:12)  '_' num2str(thlim(1)) '_' num2str(thlim(2)) '_DT_' num2str(Ith) '_TR'];
% print([outdir fout '.jpg'],'-djpeg','-r250')

%% JOINT BEHAVIOUR Hs/Tp
% reference Tr
Tr_plt = [1 25 100 1000];

figure
xline(He(Tr_plt),'g-');hold on
scatter(H,T,'o','Filled',...
    'MarkerFaceColor',[.7 .7 .7],'MarkerEdgeColor','none','MarkerFaceAlpha',.4);hold on
dscatter(Hgp,Tgp)

xlabel('H_s [m]')
ylabel('T_p [s]')

for i = 1:length(Tr_plt)
    text(He(Tr_plt(i))+.1,0.2,[num2str(Tr_plt(i)) 'y'],'Rotation',90,...
        'Color','g')
end

set(gca,'FontSize',12)

fout  = [filein(7:12)  '_' num2str(thlim(1)) '_' num2str(thlim(2)) '_DT_' num2str(Ith) '_HsTp'];
print([outdir fout '.jpg'],'-djpeg','-r250')

