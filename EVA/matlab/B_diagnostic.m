clearvars
close all
clc

addpath ./auxiliary/

indir  = './input/';
outdir = './output/'; 

Ith    = 24;            % window's width [hr]
thlim  = [180 247.5];   % directional fetch 

%% 1. load data
filein = 'Point_000230_out.dat';
[date,Hs,~,Tp,~,thp,~,~,~,~,~,~,Ny] = loadDICCA([indir filein]);

%% 2. select peaks through mov. windows and for selected fetch
[D,H,~,TH] = mov_wnd_peak(date,Hs,Tp,thp,Ith);

fout  = [filein(7:12)  '_' num2str(thlim(1)) '-' num2str(thlim(2)) '_DT_' num2str(Ith)];

fig=figure;
set(fig,'Position',[360 178 860 360])
plot(TH,H,'o','Color',[.7 .7 .7]);hold on

D  = D(TH>=thlim(1) & TH<thlim(2));
H  = H(TH>=thlim(1) & TH<thlim(2));
TH = TH(TH>=thlim(1) & TH<thlim(2));

plot(TH,H,'ko');hold on

xline([22.5:45:360],'k--')
xline([0:45:360],'k:')

xlabel('\theta_p [°N]')
ylabel('H_s [m]')
xticks([22.5:45:360])
xlim([0 360])

set(gca,'FontSize',12)
print([outdir fout '.jpg'],'-djpeg','-r250')

%% 3. sensitivity analysis 1 --> MRL
qth    = [0.1:0.05:0.95 0.96:0.01:0.99]; % quantiles for threshold testing
hth    = quantile(H,qth);
mrl    = nan(length(hth),1);lambda = nan(length(hth),1);ci = nan(length(hth),2);

for i = 1:length(hth)    
    
    u         = hth(i);
    datafit   = H(H>u) - u;
    mrl(i)    = mean(datafit);
    lambda(i) = length(datafit)/Ny; 

    % compute 95% CI
    s = std(datafit);
    ci(i,1) = mean(datafit) - 1.96*s/sqrt(length(datafit)); 
    ci(i,2) = mean(datafit) + 1.96*s/sqrt(length(datafit));    

end

rng      = [0:5:100];
uplim    = abs(rng - floor(max(lambda)));
[~,imin] = min(uplim);
uplim    = rng(imin);
nclass   = uplim/5;

lbl = [num2str(thlim(1)) '°N - ' num2str(thlim(2)) '°N'];

fout  = [filein(7:12)  '_' num2str(thlim(1)) '_' num2str(thlim(2)) '_DT_' num2str(Ith) '_MRL'];

fig=figure;
plot(hth,mrl,'k-');hold on
plot(hth,ci','k--');hold on

grid on
scatter(hth,mrl,50,lambda,'Filled','MarkerEdgeColor','k');

colormap(jet(nclass))
h=colorbar;
ylabel(h,'\lambda [#/anno]')
clim([0 uplim])

ylabel('MRL [m]')
xlabel('u [m]')

text(0.05,0.05,lbl,'Units','Normalized')

set(gca,'FontSize',12)
print([outdir fout '.jpg'],'-djpeg','-r250')

%% 4. sensitivity analysis 1 --> GOF

qth    = [0.50 0.75 0.90 0.95 0.98 0.99]; % quantiles for threshold testing
hth    = quantile(H,qth);

fig=figure;
set(fig,'Position',[360 78 560*1.5 420*1.5])
for i = 1:length(hth)    

    % select threshold exceedances
    u         = hth(i);
    datafit   = H(H>u) - u;    

    % fit GPD
    pargp = gpfit(datafit);
    sh    = pargp(1); sc = pargp(2);       

    n    = length(datafit);
    prob = [1:n]./(n+1);
    
    thr_q = gpinv(prob, sh, sc, u);  % theoretical distribution
    emp_q = sort(datafit) + u;       % empirical distribution

    subplot(2,3,i)

    plot(thr_q,emp_q,'o');hold on
    plot([-10 10],[-10 10],'--','Color',[.7 .7 .7],'LineWidth',1.2)

    axis equal
    axis([0 10 0 10])

    if i == 4 | i == 5 | i == 6
        xlabel('theoretical H_s [m]')
    end

    if i == 1 | i == 4 
        ylabel('empirical H_s [m]')
    end    

    text(0.05,0.9,['H_{th}=' num2str(u,2) 'm'],'Units','Normalized');hold on
    text(0.05,0.75,['\lambda=' num2str(n/Ny,2)],'Units','Normalized');

    set(gca,'FontSize',12)

end

fout  = [filein(7:12)  '_' num2str(thlim(1)) '_' num2str(thlim(2)) '_DT_' num2str(Ith) '_QQ'];
print([outdir fout '.jpg'],'-djpeg','-r250')