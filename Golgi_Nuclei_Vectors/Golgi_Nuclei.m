%Golgi-nuclei analysis, Golgi at the front of the migration, nuclei from
% stardist, golgi from trackmate
% Load Golgi and Nuclei data
clearvars -except LastFolder;
if exist('LastFolder','var')
    GetFileName=sprintf('%s/*.csv',LastFolder);
else
    GetFileName='*.csv';
end
[FileNameL,PathNameL] = uigetfile(GetFileName,'Select Golgi');
GetFileName=sprintf('%s/*.csv',PathNameL);
[FileNameR,PathNameR] = uigetfile(GetFileName,'Select Nuclei');
LastFolder=PathNameR;

LeftFile =sprintf('%s%s',PathNameL,FileNameL);
RightFile =sprintf('%s%s',PathNameR,FileNameR);
golgiData = readtable(LeftFile,"NumHeaderLines",3);
nucleiData = readtable(RightFile,"NumHeaderLines",3);

% Extract x and y positions
golgiPositions = [golgiData.x_inch_, -golgiData.x_inch__1];
nucleiPositions = [nucleiData.x_inch_, -nucleiData.x_inch__1];
% Remove rows with 0 or -1 in x or y positions for Golgi
golgiPositions = golgiPositions(~any(golgiPositions == 0 | golgiPositions == -1 | golgiPositions == 1, 2), :);
nucleiPositions = nucleiPositions(~any(nucleiPositions == 0 | nucleiPositions == -1 | nucleiPositions == 1, 2), :);
% Calculate the center of the image
centerPos = mean([golgiPositions; nucleiPositions]);
% Initialize the cost matrix
numGolgis = size(golgiPositions, 1);
numNuclei = size(nucleiPositions, 1);
costMatrix = zeros(numGolgis, numNuclei);

% Calculate distances to populate the cost matrix
for i = 1:numGolgis
    for j = 1:numNuclei
        costMatrix(i, j) = norm(golgiPositions(i, :) - nucleiPositions(j, :));
    end
end

% Use the Hungarian algorithm to find the optimal unique pairs
% `matchpairs` minimizes the cost for one-to-one assignments.
costUnmatched = max(size(costMatrix)) * max(costMatrix,[],'all');
[pairs, ~] = matchpairs(costMatrix, 1e5);         %The value can be adjusted according to the data

% Initialize a figure
figure;
hold on;

%Plot Golgi and Nuclei positions
plot(golgiPositions(:, 1), golgiPositions(:, 2), 'ro', 'MarkerSize',3, 'MarkerFaceColor', 'r', 'DisplayName', 'Golgi');
plot(nucleiPositions(:, 1), nucleiPositions(:, 2),'bo', 'MarkerSize',3,'MarkerFaceColor', 'b', 'DisplayName', 'Nuclei');

%For each Golgi position, find the nearest Nucleus and draw an arrow
%Plot arrows for each unique pair
%Prepare an empty array to store angles (only for valid pairs)
angles = [];
distances = [];

% For each pair, calculate the direction vector, nucleus-to-center vector, and angle
for k = 1:size(pairs, 1)
    golgiIdx = pairs(k, 1);
    nucleusIdx = pairs(k, 2);
    
    % Get the Golgi and nucleus positions for the matched pair
    golgiPos = golgiPositions(golgiIdx, :);
    nucleusPos = nucleiPositions(nucleusIdx, :);
    % Calculate the distance between the nucleus and Golgi
    distance = norm(golgiPos - nucleusPos);
    
    % Filter out pairs with distances greater than 6 microns
    if distance > 6
        continue;
    end
    % Calculate direction vector from nucleus to Golgi
    direction = golgiPos - nucleusPos;
    direction = direction / norm(direction);  % Normalize direction vector
    
    % Calculate the vector from nucleus to the center
    nucleusToCenter = centerPos - nucleusPos;
    nucleusToCenter = nucleusToCenter / norm(nucleusToCenter);  % Use this for bead center reference of 0 degree
    nucleusToCenter = [-1,0];                                   % Use this for vertical line reference of 0 degree

    % Calculate the angle between the direction vector and nucleus-to-center vector
    angle = acosd(dot(direction, nucleusToCenter));
    % Store only valid angles
    angles = [angles; angle];
    distances = [distances; distance];

    % Plot the arrow with a color based on the angle
    q=quiver(nucleusPos(1), nucleusPos(2), ...
           direction(1) * norm(golgiPos - nucleusPos), direction(2) * norm(golgiPos - nucleusPos), ...
           1, 'LineWidth',1.5,'MaxHeadSize', 25, 'AutoScale', 'off', ...
           'Color', [1 1 1] * angle / max(angles));  % Color gradient from 0 to max angle
end
q.MaxHeadSize = 150;
hold off;

% Plot a histogram of the angles
figure;
histogram(angles, 18);  % Adjust the number of bins as needed
xlabel('Angle (degrees)');
ylabel('Frequency');
title('Histogram of Angles between Direction and Nucleus-to-Center Vector');
median(angles)

% figure;
% scatter(angles,distances,'Marker','.');
% xlabel('Angle (degrees)');
% ylabel('Distance (um)');     
% [rho,pval] = corrcoef(angles,distances)