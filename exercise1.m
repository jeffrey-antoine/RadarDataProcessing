% This script has been varified with other linear kalman filter
% Jileil @Hasco 2018-03-23
clc;clear;close all;
load trueTarget.mat
% trueTarget = zeros(3,151);
% trueTarget(2:3,:) = [1000:10:2500;1000*ones(1,151)];
%% generate noisyMeasurement variable

% this section generates random noisy measurement variable
% set the covariance matrix
% Sigma = [SigmaX^2 0;
%          0 SigmaY^2]
sigma = 100;
% 因为噪声是相互独立的，因此其非对角元素必为0
Sigma = [sigma^2 0;0 sigma^2]; 
% Initialize noisyMeasurement
noisyMeasurement = zeros(size(trueTarget));
noisyMeasurement(1,:) = trueTarget(1,:);
for i = 1:size(trueTarget,2)
    % Generate pseudo trueTarget
   
    %randn is the random normal ditribution generator
    % get cholesky decomposition A = LL', L is a upper triangle matrix
    niu = chol(Sigma)*randn(2,1);
    %Znoisy = Ztrue + Noisy
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
% 过程噪声模型
% process noise
% 表征了系统模型的统计特性，增加Q中元素的值，等价于增加系统噪声或增加系统
% 参数的不确定性，从而使得增益矩阵Gf增大，加大了系统校正权值，提高了系统动
% 态性能和稳态值。
pnsigma = 1;
processNoiseSigma = [pnsigma^2 0;0 pnsigma^2];
% this is Q
% noiseVector = chol(eye(2))*randn(2,1);
% 过程噪声是预测过程中混入的噪声
Q = B * processNoiseSigma * B';%[4*4] 过程噪声在高斯模型下是一个常量

H =  [1 0 0 0;
      0 1 0 0];
% 
xEstimate = zeros(4,151);
pEstimate = zeros(16,151);
xPredict = zeros(4,151);
zPredict = zeros(2,151);
innovation = zeros(2,151);
% 初始估计= z0 + chol(p0_v)*rand(2,1)
a0 = [Z(:,1);chol([100 0;0 100])*rand(2,1)];
xEstimate(:,1) = a0;
pEstimate(:,1) = reshape(P0,16,1);
% 假设未知量测噪声的分布
% 设量测噪声的协方差为sigma2 =[50^2 0;0 50^2];
sigma2 = 100;
Sigma2 = [sigma2^2 0;0 sigma2^2];
% % 假设的量测噪声模型
% w = chol(Sigma2)*randn(2,1);
for k = 2:151
    xPredict(:,k) = A * xEstimate(:,k - 1); % 1-step-ahead vector of state forecasts
    pPredict = A * reshape(pEstimate(: , k - 1),4 , 4) * A.' + Q; %[4 4] % 1-step-ahead covariance
    % 1-step-ahead vector of observation forecasts
    % zPredict = E[H(k+1)*X(K+1)+W(K+1)|Z^k], due to the assumption that
    % E[W(K+1)] = 0, zPredict can be simplified as：
    zPredict(:,k) = H * xPredict(:,k); 
    % 1-step-ahead estimated of observation covariance
    % Sigma2 严格意义上应该根据量测方程前面的系数以及Sigma2 共同得到
    Spredict = H * pPredict * H' + Sigma2; %[2 2]
    K = pPredict * H'/ Spredict; %[4 2]
    % observation innovation
    innovation(:,k) = Z(:,k) - zPredict(:,k); % 新息 [2 1]
    xEstimate(:,k) = xPredict(:,k) + K * innovation(:,k); % [4 1]
    %pEstimate(:,k) = reshape(pPredict - K * Spredict * K',16 ,1); %[4 4]
    AA = eye(length(xEstimate(:,k))) - K * H;
    pEstimate_temp = AA * pPredict * AA.' + K * Sigma2 * K.';
    pEstimate(:,k) = reshape(pEstimate_temp,16,1);
end
% errorPredict_x = trueTarget(2,2:151) - xPredict(1,2:151);
% errorPredict_y = trueTarget(3,2:151) - xPredict(2,2:151);
% errorPredict = sqrt(errorPredict_x.^2 + errorPredict_y.^2);
% 
% 
% errorEstimate_x = trueTarget(2,2:151) - xEstimate(1,2:151);
% errorEstimate_y = trueTarget(3,2:151) - xEstimate(2,2:151);
% errorEstimate = sqrt(errorEstimate_x.^2 + errorEstimate_y.^2);


% subplot(1,2,1)
%axis([500 3000 200 2200]);
hold on;
plot(trueTarget(2,:),trueTarget(3,:),'b*-');

plot(Z(1,:),Z(2,:),'r>');
plot(xEstimate(1,:),xEstimate(2,:),'go-');
%title(['kn = ' num2str(kn)]);



% subplot(1,2,2)
% 
% plot(errorPredict,'r-');
% hold on;
% plot(errorEstimate,'b-');
% 
% RMS_errorPredict = sqrt(sum(errorPredict.^2)/150);
% RMS_errorEstimate = sqrt(sum(errorEstimate.^2)/150);
% 
% hold off;

% figure
% 
% plot(xEstimate(1,:),xEstimate(2,:),'ro');
% hold on;
% plot(xPredict(1,:),xPredict(2,:),'g-');
% %%************************ Demo Kalman filter***************************
% % 参数初始化
% % x0_: mean of x0, initialization of status vector
% x0_ = [1000 1000 0 0]';
% % P0 : the initialization of covariance matrix
% P0 = [10000 0 0 0;
%       0 10000 0 0;
%       0  0  100 0;
%       0  0  0 100];
% % x0 is normal distributed as N(x0;x0_,P0)
% %x0 = chol(P0)*randn(4,1) + x0_;
% %x0 = x0_;
% % Assume time interval T = 1
% T = 1;
% % ********************* CV Model **********************
% % status transfer matrix in CV(constant-velocity) model as A or F:
% % [I2 T*I2  =  [1 0 T 0
% %  02 I2 ]     0 1 0 T
% %              0 0 1 0
% %              0 0 0 1]
% F = [1 0 T 0;
%      0 1 0 T;
%      0 0 1 0;
%      0 0 0 1];
% % noisy transfer matrix in CV(constant-velocity) model as B or T:
% % [0.5*T^2*I2 = [0.5T^2  0
% %    T*I2]       0    0.5T^2
% %                 T      0
% %                 0      T]
% B = [0.5*T^2    0;
%        0     0.5*T^2;
%        T        0;
%        0        T;];
% % 过程噪声模型
% % process noise
% % 表征了系统模型的统计特性，增加Q中元素的值，等价于增加系统噪声或增加系统
% % 参数的不确定性，从而使得增益矩阵Gf增大，加大了系统校正权值，提高了系统动
% % 态性能和稳态值。
% pnsigma = 1;
% processNoiseSigma = [pnsigma^2 0;0 pnsigma^2];
% % this is Q
% % noiseVector = chol(eye(2))*randn(2,1);
% % 过程噪声是预测过程中混入的噪声
% Q = B * processNoiseSigma * B';%[4*4] 过程噪声在高斯模型下是一个常量
% 
% H =  [1 0 0 0;
%       0 1 0 0];
% % 
% xEstimate2 = zeros(4,151);
% pEstimate2 = zeros(16,151);
% y = zeros(4,151);
% % 初始估计= z0 + chol(p0_v)*rand(2,1)
% xEstimate2(:,1) = a0;
% 
% pEstimate2(:,1) = reshape(P0,16,1);
% % 假设未知量测噪声的分布
% sigma2 = 100;
% R = [sigma2^2 0;0 sigma2^2];
% h2 = figure;
% set(h2,'position',[100,100,1600,800]);
% for k = 2:151
%     x_hat = xEstimate2(:,k-1);
%     P = reshape(pEstimate2(:,k-1),4,4);
%     [x_hat,P,y0] = lkf(x_hat,P,0,Z(:,k),F,zeros(4,1),H,zeros(2,1),Q,R);
%     xEstimate2(:,k) = x_hat;
%     pEstimate2(:,k) = reshape(P,16,1);
%     y(:,k) = y0;
% end
% hold on;
% plot(trueTarget(2,:),trueTarget(3,:),'b*-');
% 
% plot(Z(1,:),Z(2,:),'r>');
% plot(xEstimate2(1,:),xEstimate2(2,:),'go-');