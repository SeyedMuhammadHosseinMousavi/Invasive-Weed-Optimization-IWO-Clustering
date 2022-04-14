% Invasive Weed Optimization (IWO) Clustering
% Created By Seyed Muhammad Hossein Mousavi - 2022
% Comparison with K-means and GMM
clc;
clear;
close all;
warning('off');

%% Basics 
% Loading
data = load('dat');
X = data.XX;
%
k = 3; % Number of Clusters
%
CostFunction=@(m) ClusterCost(m, X);     % Cost Function
VarSize=[k size(X,2)];           % Decision Variables Matrix Size
nVar=prod(VarSize);              % Number of Decision Variables
VarMin= repmat(min(X),k,1);      % Lower Bound of Variables
VarMax= repmat(max(X),k,1);      % Upper Bound of Variables

%% IWO Params
MaxIt = 25;     % Maximum Number of Iterations
nPop0 = 2;      % Initial Population Size
nPop = 5;       % Maximum Population Size
Smin = 2;       % Minimum Number of Seeds
Smax = 5;       % Maximum Number of Seeds
Exponent = 1.5;           % Variance Reduction Exponent
sigma_initial = 0.2;    % Initial Value of Standard Deviation
sigma_final = 0.001;	% Final Value of Standard Deviation

%% Intro
% Empty Plant Structure
empty_plant.Position = [];
empty_plant.Cost = [];
empty_plant.Out = [];
pop = repmat(empty_plant, nPop0, 1);    % Initial Population Array
for i = 1:numel(pop)
% Initialize Position
pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
% Evaluation
[pop(i).Cost, pop(i).Out]= CostFunction(pop(i).Position);
end
% Best Solution Ever Found
BestSol = pop(1);
% Initialize Best Cost History
BestCosts = zeros(MaxIt, 1);

%% IWO Main Body
for it = 1:MaxIt
% Update Standard Deviation
sigma = ((MaxIt - it)/(MaxIt - 1))^Exponent * (sigma_initial - sigma_final) + sigma_final;
% Get Best and Worst Cost Values
Costs = [pop.Cost];
BestCost = min(Costs);
WorstCost = max(Costs);
% Initialize Offsprings Population
newpop = [];
% Reproduction
for i = 1:numel(pop)
ratio = (pop(i).Cost - WorstCost)/(BestCost - WorstCost);
S = floor(Smin + (Smax - Smin)*ratio);
for j = 1:S
% Initialize Offspring
newsol = empty_plant;
% Generate Random Location
newsol.Position = pop(i).Position + sigma * randn(VarSize);
% Apply Lower/Upper Bounds
newsol.Position = max(newsol.Position, VarMin);
newsol.Position = min(newsol.Position, VarMax);
% Evaluate Offsring
[newsol.Cost, newsol.Out] = CostFunction(newsol.Position);
% Add Offpsring to the Population
newpop = [newpop
newsol];  
end
end
% Merge Populations
pop = [pop
newpop];
% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Competitive Exclusion (Delete Extra Members)
if numel(pop)>nPop
pop = pop(1:nPop);
end
% Store Best Solution Ever Found
BestSol = pop(1);
% Store Best Cost History
BestCosts(it) = BestSol.Cost;
% Display Iteration Information
disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCosts(it))]);
% Plot 
DECenters=PlotRes(X, BestSol);
pause(0.01);
end

%% Plot IWO Train
figure;
semilogy(BestCosts, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;

DElbl=BestSol.Out.ind;

%% K-Means Clustering for Comparison
[kidx,KCenters] = kmeans(X,k);
figure
set(gcf, 'Position',  [150, 50, 700, 400])
subplot(2,3,1)
gscatter(X(:,1),X(:,2),kidx);title('K-Means')
hold on;
plot(KCenters(:,1),KCenters(:,2),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,2)
gscatter(X(:,1),X(:,3),kidx);title('K-Means')
hold on;
plot(KCenters(:,1),KCenters(:,3),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,3)
gscatter(X(:,1),X(:,4),kidx);title('K-Means')
hold on;
plot(KCenters(:,1),KCenters(:,4),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,4)
gscatter(X(:,2),X(:,3),kidx);title('K-Means')
hold on;
plot(KCenters(:,2),KCenters(:,3),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,5)
gscatter(X(:,2),X(:,4),kidx);title('K-Means')
hold on;
plot(KCenters(:,2),KCenters(:,4),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,6)
gscatter(X(:,3),X(:,4),kidx);title('K-Means')
hold on;
plot(KCenters(:,3),KCenters(:,4),'ok','LineWidth',2,'MarkerSize',6);
%
KMeanslbl=kidx;
%% Gaussian Mixture Model Clustering for Comparison
options = statset('Display','final'); 
gm = fitgmdist(X,k,'Options',options)
idx = cluster(gm,X);
figure
set(gcf, 'Position',  [50, 300, 700, 400])
subplot(2,3,1)
gscatter(X(:,1),X(:,2),idx);title('GMM')
hold on;
subplot(2,3,2)
gscatter(X(:,1),X(:,3),idx);title('GMM')
hold on;
subplot(2,3,3)
gscatter(X(:,1),X(:,4),idx);title('GMM')
hold on;
subplot(2,3,4)
gscatter(X(:,2),X(:,3),idx);title('GMM')
hold on;
subplot(2,3,5)
gscatter(X(:,2),X(:,4),idx);title('GMM')
hold on;
subplot(2,3,6)
gscatter(X(:,3),X(:,4),idx);title('GMM')
hold on;
%
GMMlbl=idx;

%% MAE and MSE Errors
IWO_GMM_MAE=mae(DElbl,GMMlbl);
IWO_KMeans_MAE=mae(DElbl,KMeanslbl);
GMM_KMeans_MAE=mae(GMMlbl,KMeanslbl);
IWO_GMM_MSE=mse(DElbl,GMMlbl);
IWO_KMeans_MSE=mse(DElbl,KMeanslbl);
GMM_KMeans_MSE=mse(GMMlbl,KMeanslbl);
fprintf('IWO vs GMM MAE =  %0.4f.\n',IWO_GMM_MAE)
fprintf('IWO vs K-Means MAE =  %0.4f.\n',IWO_KMeans_MAE)
fprintf('GMM vs K-Means MAE =  %0.4f.\n',GMM_KMeans_MAE)
fprintf('IWO vs GMM MSE =  %0.4f.\n',IWO_GMM_MSE)
fprintf('IWO vs K-Means MSE =  %0.4f.\n',IWO_KMeans_MSE)
fprintf('GMM vs K-Means MSE =  %0.4f.\n',GMM_KMeans_MSE)


