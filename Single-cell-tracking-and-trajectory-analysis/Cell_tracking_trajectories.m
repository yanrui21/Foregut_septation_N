
% PLOT_TRACKS_IN_ROI  Draw TrackMate trajectories that start inside a square ROI.
%
%   plot_tracks_in_ROI('6_edges.csv')
%
% The script assumes TrackMate “edge” export with three header lines:
%   row 0 – column names, row 1 – aliases, row 2 – units.
% Feel free to adapt variable names if your export differs.
% For vbSPT analysis of diffusion states, please install dependencies from https://github.com/bmelinden/vbSPT

%% 1. read the CSV --------------------------------------------------------
clear all;
    csvFile = '6_edges.csv';           % change the file path

opts               = detectImportOptions(csvFile,'NumHeaderLines',3);
T                  = readtable(csvFile,opts);

% Give human-friendly names
T.Properties.VariableNames = {'Label','TrackID','SourceID' ,'TargetID','Cost','DirChange','Speed','Disp','Time','X','Y','Z' };

%% 2. user-defined ROI (square, microns) ----------------------------------
% Centre (µm) and half-size (µm) of the square ROI
roiCentre   = [288 , 180];     % [x0 , y0] in microns
roiHalfSize = [60, 60];              % half-length of a side
toiWindow   = [0 , 15];     % time window in frame
t_max = toiWindow(2) - toiWindow(1);
calib = 7.5;      % minutes per frame
xmin = roiCentre(1) - roiHalfSize(1);
xmax = roiCentre(1) + roiHalfSize(1);
ymax = 420- (roiCentre(2) - roiHalfSize(2));     % invert y axis to match MATLAB matrix with image (subtract from the image size)
ymin = 420- (roiCentre(2) + roiHalfSize(2));
T.Y = 420 - T.Y;
%% 3. gather trajectories that begin inside the ROI -----------------------
trackIDs = unique(T.TrackID);
keepID   = false(size(trackIDs));

for i = 1:numel(trackIDs)
    this      = T(T.TrackID == trackIDs(i),:);
    this      = sortrows(this,'Time');          % chronological order
    if ~isempty(toiWindow)
        this = this(this.Time >= toiWindow(1) & this.Time <= toiWindow(2), :);
    end
    if height(this) < 3,  continue;  end         % need ≥2 edges to plot
    x0        = this.X(1);
    y0        = this.Y(1);
    t0        = this.Time(1);
    keepID(i) = (x0>=xmin && x0<=xmax && y0>=ymin && y0<=ymax);
end

selectedIDs = trackIDs(keepID);

%% 4. plot ----------------------------------------------------------------
figure; hold on;
cmap = jet(256);                         % Jet LUT
for id = selectedIDs.'
    traj        = sortrows(T(T.TrackID==id,:), 'Time');
    if ~isempty(toiWindow)
        traj = traj(traj.Time >= toiWindow(1) & traj.Time <= toiWindow(2), :);
    end;
    if height(traj) < 3,  continue;  end;
    % translate so first edge is at (0,0)
    %x           = traj.X - traj.X(1);
    %y           = traj.Y - traj.Y(1);
    x           = traj.X;
    y           = traj.Y;
    % colour vector normalised 0-1 by elapsed time
    t           = traj.Time;
    cNorm       = (t - t(1)) ./ t_max;      % 0 … 1
    cIdx        = round( 1 + cNorm * (size(cmap,1)-1) );

    % draw each edge segment in its colour
    for k = 1:numel(x)-1
        plot( x(k:k+1), y(k:k+1), ...
              'Color', cmap(cIdx(k),:), 'LineWidth', 1.5 );
    end
end

% draw ROI outline at origin for reference (dashed grey square)
% plot( [-roiHalfSize roiHalfSize roiHalfSize -roiHalfSize -roiHalfSize], ...
%       [-roiHalfSize -roiHalfSize roiHalfSize roiHalfSize -roiHalfSize], ...
%       '--', 'Color', [0.5 0.5 0.5] );

axis equal
xlabel('X (µm, origin = trajectory start)')
ylabel('Y (µm, origin = trajectory start)')
title(sprintf('Trajectories starting inside [%g ± %g, %g ± %g] µm ROI', ...
      roiCentre(1), roiHalfSize, roiCentre(2), roiHalfSize))
colormap(cmap); 
% colorbar('Ticks',[0 1], ...
%                          'TickLabels',{'start','end'}, ...
%                          'Label','Normalised trajectory time')

box on; grid on;

%% -----------------------------------------------------------------------
%  SECTION 2 - Angular distribution of end-point vectors
% ------------------------------------------------------------------------

% Collect end-point angles (radians) – positive X axis = 0 rad, CCW ↑
angles = zeros(numel(selectedIDs),1);

for n = 1:numel(selectedIDs)
    traj     = sortrows(T(T.TrackID==selectedIDs(n),:),'Time');
    dx       = traj.X(end) - traj.X(1);     % displacement in µm
    dy       = traj.Y(end) - traj.Y(1);
    angles(n)= atan2(dy,dx);                % range −π … π
end

%% A. rose plot / polar histogram ----------------------------------------
figure;
polarhistogram(angles, 12, ...     % 18 bins = 20° each – can be tuned
               'FaceAlpha',0.75, ...
               'EdgeColor','none');
title('Angular distribution of final displacement vectors');
% Add radial grid label
rlim auto;

%% B. (optional) Cartesian histogram for exact counts --------------------
% Convert to degrees 0–360 for readability
angDeg = mod(rad2deg(angles), 360);

figure;
histogram(angDeg, 12, 'Normalization','probability');
xlabel('Angle (°)  – 0° = +X, 90° = +Y');
ylabel('Fraction of trajectories');
title('Angular distribution (Cartesian view)');
box on; grid on;

%% -----------------------------------------------------------------------
%  SECTION 3 – Average-velocity arrows (colour = direction)
% ------------------------------------------------------------------------
%  For every trajectory that survived the ROI + TOI filters:
%    • compute mean velocity vector  v̄  (µm s-1)
%    • draw one arrow that starts at the trajectory origin (0,0)
%    • colour the arrow by its angle using the Jet LUT
%
%  The arrow length is proportional to |v̄|, so faster tracks look longer.
% ------------------------------------------------------------------------

figure; hold on; axis equal
xlabel('X (µm, origin = first edge)')
ylabel('Y (µm, origin = first edge)')
title('Average velocity vectors (arrows coloured by direction)')

% Use the same LUT resolution you prefer elsewhere
cmap    = phasemap(256);

for id = selectedIDs.'
    % ----------------- grab the trajectory fragment ---------------------
    traj = sortrows(T(T.TrackID==id,:), 'Time');
    if ~isempty(toiWindow)
        traj = traj(traj.Time >= toiWindow(1) & traj.Time <= toiWindow(2), :);
    end
    if height(traj) < 3,  continue;  end    % need ≥2 points

    % ----------------- mean velocity components (µm s-1) ----------------
    dx   = traj.X(end) - traj.X(1);
    dy   = traj.Y(end) - traj.Y(1);
    dt   = traj.Time(end) - traj.Time(1);          % same units as Time col.
    vx   = dx / dt;
    vy   = dy / dt;
    x0 = traj.X(1);                     % start-of-track X (µm)
    y0 = traj.Y(1);                     % start-of-track Y (µm)
    % ----------------- colour by angle ----------------------------------
    ang  = atan2(vy, vx);                          % −π … π
    cIdx = round( 1 + mod(ang - pi, 2*pi) / (2*pi) * (size(cmap,1)-1) );
    thisColour = cmap(cIdx,:);

    % ----------------- draw arrow at origin -----------------------------
    quiver(x0, y0, vx, vy, 6,...          % <-- (x0,y0) not (0,0)
       'Color', thisColour, ...
       'LineWidth', 1.8, ...
       'MaxHeadSize', 20, ...
       'AutoScale', 'off');
end

% optional: grey square showing ± roiHalfSize for context
% plot([-roiHalfSize roiHalfSize roiHalfSize -roiHalfSize -roiHalfSize], ...
%      [-roiHalfSize -roiHalfSize roiHalfSize roiHalfSize -roiHalfSize], ...
%      '--', 'Color', [0.6 0.6 0.6]);

% ----------------- colourbar keyed to angle -----------------------------
cb = colorbar;
set(cb, 'Ticks', [0 0.25 0.5 0.75 1], ...
        'TickLabels', {'0°', '90°', '180°', '270°', '360°'});
colormap(cmap);
cb.Label.String = 'Arrow colour encodes direction (deg)';

box on; grid on;

%% -----------------------------------------------------------------------
%  SECTION 4 – MSD curves + grand mean ± SD
% ------------------------------------------------------------------------
figure; hold on;
title('Time-averaged MSD of each trajectory');
xlabel('Frame lag  m');              % multiply by Δt later if desired
ylabel('\langle (\Delta r)^2 \rangle  (\mum^2)');

% ---------- FIRST PASS: compute & store each MSD ------------------------
allMSD   = {};                       % will become {nTracks × 1} cell array
maxLagGl = 0;                        % longest lag among all tracks

for id = selectedIDs.'
    traj = sortrows(T(T.TrackID==id,:), 'Time');
    if ~isempty(toiWindow)
        traj = traj(traj.Time >= toiWindow(1) & traj.Time <= toiWindow(2), :);
    end
    N = height(traj);
    if N < 3,  continue;  end

    x = traj.X;  y = traj.Y;
    msd = zeros(N-1,1);
    for m = 1:(N-1)
        dx = x(1+m:end) - x(1:end-m);
        dy = y(1+m:end) - y(1:end-m);
        msd(m) = mean(dx.^2 + dy.^2);
    end

    % ----- plot individual curve in a random colour
    plot(1:(N-1), msd, 'Color', rand(1,3), 'LineWidth', 1.2);

    % ----- store for ensemble stats
    allMSD{end+1,1} = msd;          
    maxLagGl        = max(maxLagGl, numel(msd));
end

% ---------- SECOND PASS: assemble to rectangular array with NaNs ---------
nTracks = numel(allMSD);
msdMat  = NaN(nTracks, maxLagGl);    % rows = tracks, cols = lag
for k = 1:nTracks
    msdMat(k,1:numel(allMSD{k})) = allMSD{k};
end

% ---------- MEAN and SD across trajectories -----------------------------
meanMSD = nanmean(msdMat, 1);        % ignore NaNs (short tracks)
sdMSD   = nanstd(msdMat, 0, 1);

lags    = 1:maxLagGl;

% ---------- plot mean ± 1 SD (black curve, grey error) ------------------
% Shaded patch for SD
patch([lags fliplr(lags)], ...
      [meanMSD+sdMSD  fliplr(meanMSD-sdMSD)], ...
      [0.7 0.7 0.7], 'FaceAlpha',0.3, 'EdgeColor','none');

% Mean MSD in bold black
plot(lags, meanMSD, 'k', 'LineWidth', 2.5);

% ---------- cosmetics ----------------------------------------------------
legend({'individual tracks', '±1 SD', 'mean MSD'}, 'Location','northwest');
box on; grid on;

% ---------- POWER-LAW fit  < MSD >  = k · t^alpha -----------------------
% Use only lags where the mean curve is finite & positive
goodIdx = isfinite(meanMSD) & meanMSD > 0;
tFit    = lags(goodIdx)';            % independent variable (frame lag)
yFit    = meanMSD(goodIdx)';         % dependent variable (µm^2)

% Linear regression in log–log space:  log y = log k  + alpha · log t
p       = polyfit(log(tFit), log(yFit), 1);
alpha   = p(1);                      % slope  = exponent
k       = exp(p(2));                 % intercept -> k

% Overlay fit curve
yModel  = k * tFit.^alpha;
plot(tFit, yModel, 'r--', 'LineWidth', 2);

% Annotate the graph
text(tFit(end)*0.6, yModel(end)*1.1, ...
     sprintf('\\itk\\rm = %.3g  |  \\alpha = %.3f', k, alpha), ...
     'Color','r', 'FontSize',10, 'FontWeight','bold');

legend({'individual tracks', '±1 SD', 'mean MSD', 'power-law fit'}, ...
       'Location','northwest');

% -------- console output -----------------------------------------------
fprintf('Power-law MSD fit:\n    k      = %.6g (µm^2)\n    alpha  = %.4f\n', k, alpha);

%% -----------------------------------------------------------------------
%  SECTION 5 – Directionality & Persistence
% ------------------------------------------------------------------------
%
%  Persistence model (2-D correlated random walk):
%      <cos φ> = R     where φ = turning angle between successive steps
%      L_p   =   l̄ / (1 – R)            (µm)     – spatial persistence
%      T_p   = Δt̄ (1 + R)/(1 – R)       (s  )    – temporal persistence
%
%  l̄    = mean step length in µm
%  Δt̄   = mean inter-frame interval (uses first–order differences)
% ------------------------------------------------------------------------

% --------- initialise collectors ----------------------------------------
allTurn   = [];                  % turning angles (radians, –π … π)
LpTracks  = [];                  % per-track persistence lengths
TpTracks  = [];                  % per-track persistence times
maxLagGlobal = 8; 
CvMat  = NaN(numel(selectedIDs), maxLagGlobal);
CtMat  = NaN(numel(selectedIDs), maxLagGlobal);
dtAll  = [];                              % collect Δt to estimate Δt̄
row = 0;
for id = selectedIDs.'
    traj = sortrows(T(T.TrackID==id,:), 'Time');
    if ~isempty(toiWindow)
        traj = traj(traj.Time >= toiWindow(1) & traj.Time <= toiWindow(2), :);
    end
    N = height(traj);
    if N < maxLagGlobal + 2,  continue;  end     % need ≥3 positions for turning angles
    row = row + 1;
    % -- step vectors -----------------------------------------------------
    dx = diff(traj.X);
    dy = diff(traj.Y);
    dt = diff(traj.Time);
    vx = dx ./ dt;      vy = dy ./ dt;
    % -- heading angles ---------------------------------------------------
    theta = atan2(dy, dx);                      % angle of each step
    turn  = diff(unwrap(theta));                % turning angle φ_i
    v  = vx + 1i*vy;                        % complex vector for dot-product
    v2 = mean(abs(v).^2);                   % ⟨|v|²⟩  for normalisation
    % -- collect ----------------------------------------------------------
    allTurn = [allTurn; turn];                  
    for m = 1:maxLagGlobal
        CvMat(row,m) = real( mean( v(1:end-m) .* conj(v(1+m:end)) ) ) / v2;
        CtMat(row,m) = mean( cos( theta(1+m:end) - theta(1:end-m) ) );
    end
    dtAll = [dtAll; dt]; 
    % -- persistence metrics for THIS track ------------------------------
    R   = mean(cos(turn));                      % <cos φ>
    lBar= mean( sqrt(dx.^2 + dy.^2) );          % mean step length (µm)
    dtBar = mean(dt);                           % mean frame interval

    LpTracks(end+1,1) = lBar / (1 - R);      
    TpTracks(end+1,1) = dtBar * (1 + R)/(1 - R);
end
% 3. ensemble mean ± SD  (ignore NaNs from short tracks)
CvMean = nanmean(CvMat,1);  CvSD = nanstd(CvMat,0,1);
CtMean = nanmean(CtMat,1);  CtSD = nanstd(CtMat,0,1);

lags   = 1:maxLagGlobal;                   % frame lags
dtBar  = median(dtAll);                    % typical Δt  (time-units)
tau    = lags * dtBar;                     % convert to seconds if wanted
% --------- 5A  Turning-angle plots --------------------------------------
nbins = 12;                                     % 30° each

figure;
histogram(rad2deg(allTurn), nbins, 'Normalization','probability', ...
          'FaceColor',[0.2 0.6 0.8], 'EdgeColor','none');
xlabel('Turning angle  φ  (deg, CCW positive)');
ylabel('Fraction of steps');
title('Turning-angle distribution (Cartesian)'); grid on; box on;

figure;
polarhistogram(allTurn, nbins, 'Normalization','probability', ...
               'FaceAlpha',0.8, 'EdgeColor','none');
title('Turning-angle distribution (rose plot)');

% --------- 5B  Persistence summary --------------------------------------
meanLp = mean(LpTracks);   sdLp = std(LpTracks);
meanTp = mean(TpTracks);   sdTp = std(TpTracks);

fprintf('\nDirectionality & Persistence summary (n = %d tracks)\n', numel(LpTracks));
fprintf('  Persistence length  L_p  = %.3g ± %.3g  µm\n', meanLp, sdLp);
fprintf('  Persistence time    T_p  = %.3g ± %.3g  time-units\n', meanTp, sdTp);
figure; hold on;

% shaded SD envelope – velocity
fill([tau fliplr(tau)], ...
     [CvMean+CvSD fliplr(CvMean-CvSD)], ...
     [0.3 0.7 1], 'FaceAlpha',0.25, 'EdgeColor','none');

% shaded SD envelope – direction
fill([tau fliplr(tau)], ...
     [CtMean+CtSD fliplr(CtMean-CtSD)], ...
     [1 0.4 0.6], 'FaceAlpha',0.25, 'EdgeColor','none');

plot(tau, CvMean, '-','Color',[0 0.3 0.8],'LineWidth',2.2, ...
     'DisplayName','C_v(\tau)');
plot(tau, CtMean, '-','Color',[0.8 0 0.3],'LineWidth',2.2, ...
     'DisplayName','C_\theta(\tau)');

yline(0,'k:');

ylabel('Autocorrelation');
title('Velocity and Direction Autocorrelation');
legend('C_v ±1 SD','C_\theta ±1 SD','Location','northeast');
box on; grid on;

% 5. quick numeric summary in console ------------------------------------
% characterise persistence by the lag where C drops to 1/e
[~,iV] = min(abs(CvMean - exp(-1)));
[~,iT] = min(abs(CtMean - exp(-1)));
fprintf('\nAutocorrelation summary (ensemble mean)\n');
fprintf('  1/e decay of C_v :  τ ≈ %.2f  (lag %d)\n', ...
        tau(iV), lags(iV));
fprintf('  1/e decay of C_θ :  τ ≈ %.2f  (lag %d)\n', ...
        tau(iT), lags(iT));
%% -----------------------------------------------------------------------
%  SECTION 7 – Trajectory-shape metrics, one row per track
% ------------------------------------------------------------------------
%  Columns:
%    1  TrackID
%    2  tortuosity               = total path length / net displacement
%    3  fractalDim (Katz)        = log10(n) / [log10(n) + log10(d/L)]
%    4  straightness index       = net displacement / total path length
%    5  radius of gyration  R_g  = √⟨|r  - ⟨r⟩|²⟩
%    6  ⟨MSD⟩   (mean over all lags)
%    7  net displacement (µm)
%    8  total time      (Time units, after TOI filter)
% ------------------------------------------------------------------------

metrics = [];                         % will be [nTracks × 8] numeric

for id = selectedIDs.'
    traj = sortrows(T(T.TrackID==id,:), 'Time');
    if ~isempty(toiWindow)
        traj = traj(traj.Time>=toiWindow(1) & traj.Time<=toiWindow(2), :);
    end
    N = height(traj);
    if N < 3,  continue;  end

    % -------- basic geometry --------------------------------------------
    x = traj.X;  y = traj.Y;          % µm
    dx = diff(x); dy = diff(y);

    stepLen = hypot(dx, dy);
    Lpath   = sum(stepLen);           % total path length
    netDisp = hypot( x(end)-x(1), y(end)-y(1) );
    if netDisp==0,  continue;  end    % avoid divide-by-zero

    tortuosity  = Lpath / netDisp;
    straightnes = 1 / tortuosity;     % or netDisp / Lpath

    % -------- Katz fractal dimension ------------------------------------
    d      = max( hypot(x - x(1), y - y(1)) );   % maximum “radius”
    fractD = log10(N) / ( log10(N) + log10(d / Lpath) );

    % -------- radius of gyration  R_g -----------------------------------
    cx = mean(x);  cy = mean(y);
    Rg = sqrt( mean( (x-cx).^2 + (y-cy).^2 ) );

    % -------- MSD   (time-averaged, all integer lags) -------------------
    msd   = zeros(N-1,1);
    for m = 1:N-1
        ddx = x(1+m:end) - x(1:end-m);
        ddy = y(1+m:end) - y(1:end-m);
        msd(m) = mean( ddx.^2 + ddy.^2 );
    end
    meanMSD = mean(msd, 'omitnan');

    % -------- time span -------------------------------------------------
    totTime = traj.Time(end) - traj.Time(1);

    % -------- collect ---------------------------------------------------
    metrics(end+1,:) = [ id, tortuosity, fractD, straightnes, ...
                         Rg, meanMSD, netDisp, totTime ];          
end

% ---------- convert to table for readability (optional) -----------------
metricNames = ["TrackID","Tortuosity","FractalDim","Straightness", ...
               "RadiusGyr","MeanMSD","NetDisp","TotalTime"];
metricsTbl  = array2table(metrics, 'VariableNames', metricNames);

% ---------- preview & save ---------------------------------------------
disp(metricsTbl(1:min(10,height(metricsTbl)),:))      % show first 10 rows
writetable(metricsTbl, 'trajectory_shape_metrics.csv');

fprintf('\nSaved %d trajectories × %d metrics  ➜  trajectory_shape_metrics.csv\n', ...
        height(metricsTbl), width(metricsTbl)-1);
%%   vbSPT
% --- add vbSPT to the MATLAB path (one-time per session) ----------------
addpath(genpath('vbSPT-master'));     % ← adjust to your install
% ‘traj’ must be a cell array, each cell = [x y] positions (rows = frames)
traj = cell(numel(selectedIDs),1);

for n = 1:numel(selectedIDs)
    id   = selectedIDs(n);
    tr   = sortrows( T(T.TrackID==id,:), 'Time' );
    if ~isempty(toiWindow)
        tr = tr(tr.Time>=toiWindow(1) & tr.Time<=toiWindow(2),:);
    end
    N = height(tr);
    if N < 3,  continue;  end
    traj{n} = [ tr.X  tr.Y ];          % 2-D

end


save  myTraj.mat  traj  ;         % vbSPT will load this file

R = VB3_HMManalysis( 'runinput_Rui.m' );   % core inference call
VB3_getResult      ( 'runinput_Rui.m' );    % quick numerical summary
VB3_displayHMMmodel( 'runinput_Rui.m' );    % nice diffusion-tree figure
load  myTraj_vbSPT.mat   Wbest  options     % the key structs

K       = Wbest.N;                  % number of diffusive states
D_state = Wbest.est.DdtMean / options.timestep;  % µm^2/s
A       = Wbest.est.Amean;          % K×K transition matrix (prob / frame)

% Viterbi paths (if stateEstimate==true)
stateSeq = Wbest.est2.sMaxP;        % {nTracks×1} cell array, each = [steps×1]

fprintf('vbSPT found %d states with D = %s µm^2/s\n', ...
        K, mat2str(D_state,3));
cmapVB = lines(K);  figure; hold on; axis equal
for n = 1:numel(selectedIDs)-1
    id  = selectedIDs(n);
    tr  = traj{n};
    st  = stateSeq{n};
    for k = 1:numel(st)-1
        plot( tr(k:k+1,1), tr(k:k+1,2), ...
              'Color', cmapVB(st(k),:), 'LineWidth',1.2 );
    end
end
title(sprintf('Trajectories coloured by vbSPT state (K = %d)',K));
