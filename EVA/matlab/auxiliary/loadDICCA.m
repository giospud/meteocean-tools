function [date,Hs,Tm,Tp,thm,thp,ths,dep,Lm,Lp,uw,vw,Ny] = loadDICCA(fileload)

% upper limits to get rid of spike values
Htop = 10;
Ttop = 15;
Ltop = 1000;

% Read and parse date file
fid = fopen(fileload,'rt');
dv = textscan(fid, ['%d %d %d %d:%d:%d' repmat('%d',1,11) ],...
    'CollectOutput',true);
fclose(fid);

ddate = double(cell2mat(dv));
date = datetime(ddate(:,1),ddate(:,2),ddate(:,3),ddate(:,4),ddate(:,5),...
    ddate(:,6));

% data loading
D = load(fileload);
Hs = D(:,5);            % significant wave height
Tm = D(:,6);            % mean period
Tp = D(:,7);            % peak period
thm = D(:,8);           % mean direction (nautical convention)
thp = D(:,9);           % peak direction
ths = D(:,10);          % directional spreading
dep = D(:,11);          % water depth
Lm = D(:,12);           % mean wave length
Lp = D(:,13);           % peak wave length
uw = D(:,14);           % wind speed (zonal component)
vw = D(:,15);           % wind speed (meridional component)

% get rid of freaky data
[~,indH]  = despike(Hs,Htop);
[~,indT1] = despike(Tm,Ttop);
[~,indT2] = despike(Tp,Ttop);
[~,indL1] = despike(Lm,Ltop);
[~,indL2] = despike(Lp,Ltop);

indFREAK = unique([indH indT1 indT2 indL1 indL2]);       
    
if isempty(indFREAK)==0
    
    date(indFREAK) = [];
    Hs(indFREAK)   = [];
    Tm(indFREAK)   = [];
    Tp(indFREAK)   = [];
    thm(indFREAK)  = [];
    thp(indFREAK)  = [];
    ths(indFREAK)  = [];
    dep(indFREAK)  = [];
    Lm(indFREAK)   = [];
    Lp(indFREAK)   = [];
    uw(indFREAK)   = [];
    vw(indFREAK)   = [];
    
end

% number of years
Ny = year(date(numel(date)))-year(date(1)) + 1;

end