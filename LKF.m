% Linear Kalman Filter Demo
% Jileil  2018-03-23
clc;clear;close all;
load trueTarget.mat
% All rights reserved
%% generate noisyMeasurement variable
% this section generates random noisy measurement variable
% set the covariance matrix
% Assume noise of X and Y are independent, so the non-diagnoal elements are zeros
% Sigma = [SigmaX^2 0;
%          0 SigmaY^2]
sigma = 100;
Sigma = [sigma^2 0;0 sigma^2]; 
% Initialize noisyMeasurement
noisyMeasurement = zeros(size(trueTarget));
noisyMeasurement(1,:) = trueTarget(1,:);
for i = 1:size(trueTarget,2)
    % Generate pseudo trueTarget 
    %randn is the random normal ditribution generator
    % get cholesky decomposition A = LL', L is a upper triangle matrix in matlab
    niu = chol(Sigma)*randn(2,1);
    % Znoisy = Ztrue + Noisy
    % Generate Noise Measurement 2
    noisyMeasurement(2:3,i) = trueTarget(2:3,i) + niu;
end
Z = noisyMeasurement(2:3,:);
clear Sigma sigma niu i;
%% Kalman Fliter implementation
h = figure;
set(h,'position',[100,100,800,800]);

% x0_: mean of x0, initialization of status vector
x0_ = [1000 1000 0 0]';
% P0 : the initialization of covariance matrix
P0 = [10000 0 0 0;
      0 10000 0 0;
      0  0  100 0;
      0  0  0 100];
% x0 is normal distributed as N(x0;x0_,P0)
%x0 = chol(P0)*randn(4,1) + x0_;
%x0 = x0_;
% Assume time interval T = 1
T = 1;
% ********************* CV Model **********************
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
% process noise model
% the dynamic property or the instability of system increases with Q;
% when Q increases, the K gets bigger
pnsigma = 1;
processNoiseSigma = [pnsigma^2 0;0 pnsigma^2];
% this is Q
% noiseVector = chol(eye(2))*randn(2,1);
Q = B * processNoiseSigma * B';%[4*4] which is a constant under the Gaussian Assumption

H =  [1 0 0 0;
      0 1 0 0];
% 
xEstimate = zeros(4,151);
pEstimate = zeros(16,151);
xPredict = zeros(4,151);
zPredict = zeros(2,151);
innovation = zeros(2,151);
% initial_estimate= z0 + chol(p0_v)*rand(2,1)
a0 = [Z(:,1);chol([100 0;0 100])*rand(2,1)];
xEstimate(:,1) = a0;
pEstimate(:,1) = reshape(P0,16,1);
% under real circumstance, we don't know the pdf of measurement noise
sigma2 = 100;
R = [sigma2^2 0;0 sigma2^2];
for k = 2:151
    xPredict(:,k) = A * xEstimate(:,k - 1); % 1-step-ahead vector of state forecasts
    pPredict = A * reshape(pEstimate(: , k - 1),4 , 4) * A.' + Q; %[4 4] % 1-step-ahead covariance
    % 1-step-ahead vector of observation forecasts
    % zPredict = E[H(k+1)*X(K+1)+W(K+1)|Z^k], due to the assumption that
    % E[W(K+1)] = 0, zPredict can be simplified as:
    zPredict(:,k) = H * xPredict(:,k); 
    % 1-step-ahead estimated of observation covariance
    % R actually need to be calculated as Q
    Spredict = H * pPredict * H' + R; %[2 2]
    K = pPredict * H'/ Spredict; %[4 2]
    % observation innovation
    innovation(:,k) = Z(:,k) - zPredict(:,k); % innovation [2 1]
    xEstimate(:,k) = xPredict(:,k) + K * innovation(:,k); % [4 1]
    %pEstimate(:,k) = reshape(pPredict - K * Spredict * K',16 ,1); %[4 4]
    AA = eye(length(xEstimate(:,k))) - K * H;
    pEstimate_temp = AA * pPredict * AA.' + K * R * K.';
    pEstimate(:,k) = reshape(pEstimate_temp,16,1);
end
hold on;
plot(trueTarget(2,:),trueTarget(3,:),'b*-');

plot(Z(1,:),Z(2,:),'r>');
plot(xEstimate(1,:),xEstimate(2,:),'go-');
