% exercise1_part2
% Jileil 2018-03-26
clc;clear;close all;
load trueTarget.mat;
rng(7);
%% UKF
% generate noisy measurement in non-linear coordinate
sigma_r = 100;
sigma_theta = 5;
% [   V_r(k)    ~ N([0 ,[sigma_r^2     0      
%  V_theta(k)]       0]      0    sigma_theta^2]
r = sqrt(trueTarget(2,:).^2 + trueTarget(3,:).^2);
theta = atan(trueTarget(3,:)./trueTarget(2,:))/pi*180;
%z_rho_theta =
Z_degree = [r;theta];

for i = 1:length(r)
    Z_degree(:,i) = Z_degree(:,i) + [100 0;0 5]*randn(2,1);
end

%%
%****************Assume Only measurement equation is different***********

% x0_: mean of x0, initialization of status vector
x0_ = [1000 1000 0 0]';
% P0 : the initialization of covariance matrix
P0 = [10000 0 0 0;
      0 10000 0 0;
      0  0  100 0;
      0  0  0 100];

% Assume time interval T = 1
T = 1;
% status transfer matrix in CV(constant-velocity) model as A or F:
% [I2 T*I2  =  [1 0 T 0
%  02 I2 ]     0 1 0 T
%              0 0 1 0
%              0 0 0 1]
A = [1 0 T 0;
     0 1 0 T;
     0 0 1 0;
     0 0 0 1];
% noisy transfer matrix in CV(constant-velocity) model as B or T:
% [0.5*T^2*I2 = [0.5T^2  0
%    T*I2]       0    0.5T^2
%                 T      0
%                 0      T]
B = [0.5*T^2    0;
       0     0.5*T^2;
       T        0;
       0        T;];
pnsigma = 1;
processNoiseSigma = [pnsigma^2 0;0 pnsigma^2];
% this is Q
% noiseVector = chol(eye(2))*randn(2,1);
Q = B * processNoiseSigma * B';%[4*4]
%*******************basic UKF*****************
n = 4;
L = 2 * n + 1;
xPredict_linear_hat = zeros(4,151);
xPredict_sumSigma_hat = zeros(4,151);
xEstimate = zeros(4,151);
% Initialize x0, p0
xEstimate(1:2,1) = [1000,1000];
pEstimate = zeros(16,151);

R = [100^2 0;0 5^2];

pPredict_sigmaPoint = zeros(L*16,151); % for each sigma Point, there is a P
zPredict_sigmaPoint = zeros(2,L); % for each sigma point, there is a measurement predict
xPredict_sigmaPoint = zeros(4,L);  % 4, L
delta_x = zeros(1,n);
w0 = 1/9;
wi = (1 - w0)/(2*n); % for this case, wi = w0
%%
for k = 2:151
    % estimation is the same as LKF
    pEstimate_reshape = reshape(pEstimate(:,k - 1),4,4);
    xPredict_linear_hat(:,k) = A * xEstimate(:,k - 1); % 1-step-ahead vector of state forecasts
    pPredict_linear = A * reshape(pEstimate(: , k - 1),4 , 4) * A.' + Q; %[4 4] % 1-step-ahead covariance
    % generate sigma point
    [u,s] = svd(pPredict_linear);
    xPredict_sigmaPoint(:,L) = xPredict_linear_hat(:,k); % x0 is placed at the end of array
    delta_x = sqrt(n/(1-w0)) * u * sqrt(s); % get the square root of pEstimate and remain the same dimension
    
    %
    for i = 1:n
         % use basic 
        xPredict_sigmaPoint(:,i) = xPredict_linear_hat(:,k) + delta_x(:,i);
        xPredict_sigmaPoint(:,i + n) = xPredict_linear_hat(:,k) - delta_x(:,i);
    end
    
    % measurement_prediction
    zPredict_sigmaPoint(1,:) = sqrt(xPredict_sigmaPoint(1,:).^2 + xPredict_sigmaPoint(2,:).^2);
    zPredict_sigmaPoint(2,:) = atan(xPredict_sigmaPoint(2,:)./xPredict_sigmaPoint(1,:))/pi*180;% radius to degree
    % wi = w0 in this example
    zPredict = wi * sum(zPredict_sigmaPoint,2);
    zll = zPredict_sigmaPoint - repmat(zPredict,1,L);
    xll = xPredict_sigmaPoint - repmat(xPredict_linear_hat(:,k),1,L);
    P_zz =  R + wi * (zll * zll');
    P_xz = wi * (xll * zll');
    % kalman gain
    K = P_xz / P_zz; 
    % correct the state
    xEstimate(:,k) = xPredict_linear_hat(:,k) + K * (Z_degree(:,k) - zPredict);
    % correct the covariance
    pEstimate_temp = pPredict_linear - K * P_zz * K';
    pEstimate(:,k) = reshape(pEstimate_temp,16,1);
end

x_rd(1,:) = sqrt(xEstimate(1,:).^2 + xEstimate(2,:).^2);
x_rd(2,:) = atan(xEstimate(2,:)./xEstimate(1,:));
trueTarget_rd(1,:) = sqrt(trueTarget(2,:).^2 + trueTarget(3,:).^2);
trueTarget_rd(2,:) = atan(trueTarget(3,:)./trueTarget(2,:));
Z_radius = [Z_degree(1,:);Z_degree(2,:)*pi/180];
h = figure;
set(h,'position',[100 100 800 800]);

polarplot(Z_radius(2,:),Z_radius(1,:),'r-o');
hold on;
polarplot(x_rd(2,:),x_rd(1,:),'g-*');
polarplot(trueTarget_rd(2,:),trueTarget_rd(1,:),'b->');
thetalim([0 90]);
rlim([1000 3500]);
