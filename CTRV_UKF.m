
clc
clear;
figure;
set(gcf,'position',[50,50,800,800]);
%Generate fake data
t = 0:0.03:30;
R = 200;
xfinal = zeros(length(t)+ 2 * 500,1);
yfinal = zeros(length(t)+ 2 * 500,1);
x = R * cos(t * 2 * pi/30 - pi/2);
y = R * sin(t * 2 * pi/30-pi/2) + R;
y(501:end) = 600 - (y(501:end) -200);
xfinal(500:-1:1) = -0.4*pi:-0.4 * pi:-200 *pi;
xfinal(1502:end) = 0.4*pi:0.4 * pi:200*pi;
xfinal(501:1501) = x;
yfinal(500:-1:1) = 0;
yfinal(1502:end) = 800;
yfinal(501:1501) = y;
% plot(xfinal,yfinal)
hold on
plot(xfinal(1),yfinal(1),'r*');
set(gca,'YLim',[-10 900]);
z = [xfinal';yfinal'];
noise = wgn(2,2001,3);
zm = z+noise;
plot(zm(1,:),zm(2,:));
hold off;
figure;
set(gcf,'position',[50,50,800,800]);
sigmaA = 1.5;
sigmaFai = 1.0;
sigmaR = 3;
% clf;
str = ['sigmaA=' num2str(sigmaA) ' sigmaFai=' num2str(sigmaFai) ' R0=' num2str(sigmaR) ' R4=' num2str(sigmaR)];
%disp([str ' is start']);
% genPic(sigmaA,sigmaFai,sigmaR);
%             close all;
%         end
%     end
% end





%additive noise version

start = 1;
z = zm
NUM_FRAMES = size(z,2);
d = 5;
x = zeros(d,NUM_FRAMES);% x,y,v,fai,fai_dot

lamda = 3-d; %
sigmaPointCollection = zeros(5,2*d+1);
sigmaPointPrediction = zeros(5,2*d+1);
P = eye(5);
% sigmaA = 3;
% sigmaFai = 0.25;
R = [sigmaR 0;0 sigmaR];
x(1,1) = z(1,1);
x(2,1) = z(2,1);
x(3,1) = 2.0;
x(4,1) = 0/57.3; %设置一个初始扰动
x(5,1) = 0.002;
str = ['sigmaA=' num2str(sigmaA) ' sigmaFai=' num2str(sigmaFai) ' R0=' num2str(R(1,1)) ' R4=' num2str(R(2,2))];
NIS = zeros(1,NUM_FRAMES);
for k = 2:NUM_FRAMES
    deltaTime = 0.03;
    fai_k = x(4,k-1);
    %QMatrix
    Qv = [0.5 * deltaTime * deltaTime * cos(fai_k) * sigmaA;
        0.5 * deltaTime * deltaTime * sin(fai_k) * sigmaA;
        deltaTime * sigmaA;
        0.5 * deltaTime * deltaTime * sigmaFai;
        deltaTime * sigmaFai;
        ];
    Q = Qv * Qv';
    %calculate sigmaPoint
    sigmaPointCollection(:,1) = x(:,k-1);
    try
        p = chol(P,'lower');
    catch
        disp(str);
        return
    end
    sigmaPointCollection(:,2:d+1)   = x(:,k-1) + sqrt(lamda + d) * p;
    sigmaPointCollection(:,d+2:end) = x(:,k-1) - sqrt(lamda + d) * p;
    %Predicted Covariance
    w = zeros(1,2*d+1);
    for i = 2:2*d+1
        w(i) = 0.5  / (lamda + d);
    end
    w(1) = lamda / (lamda + d);
    
    % 一步预测
    for j = 1:2*d+1
        fai_k = sigmaPointCollection(4,j);
        v_k = sigmaPointCollection(3,j);
        fai_k_dot = sigmaPointCollection(5,j);
        if abs(sigmaPointCollection(4,j)) > 0.0001
            sigmaPointPrediction(1,j) = sigmaPointCollection(1,j) + v_k / fai_k_dot * (sin(fai_k + fai_k_dot * deltaTime) - sin(fai_k));
            sigmaPointPrediction(2,j) = sigmaPointCollection(2,j) + v_k / fai_k_dot * (-cos(fai_k + fai_k_dot * deltaTime) + cos(fai_k));
            sigmaPointPrediction(3,j) = sigmaPointCollection(3,j);
            sigmaPointPrediction(4,j) = sigmaPointCollection(4,j) + fai_k_dot * deltaTime;
            sigmaPointPrediction(5,j) = sigmaPointCollection(5,j);
        else
            sigmaPointPrediction(1,j) = sigmaPointCollection(1,j) + v_k * (cos(fai_k) * deltaTime);
            sigmaPointPrediction(2,j) = sigmaPointCollection(2,j) + v_k * (sin(fai_k) * deltaTime);
            sigmaPointPrediction(3,j) = sigmaPointCollection(3,j);
            sigmaPointPrediction(4,j) = sigmaPointCollection(4,j);
            sigmaPointPrediction(5,j) = sigmaPointCollection(5,j);
        end
        %这里采用的是非加性噪声，所以应该考虑其非线性，也就是说，噪声也是非线性传递的一部分
        %sigmaPointPrediction(1:5,j) = sigmaPointPrediction(1:5,j) + Qv;
    end
    
    x_predict = sigmaPointPrediction * w';
    %                 if x_predict(3) < 0
    %                     x_predict(3) = -x_predict(3);
    %                 end
    covariance_predicted = zeros(d);
    x_predicted_error = sigmaPointPrediction - repmat(x_predict,[1,2*d+1]);
    for j = 1:2*d+1
        if x_predicted_error(4,j) > pi
            x_predicted_error(4,j) = x_predicted_error(4,j) - 2 * pi;
        elseif x_predicted_error(4,j) < -pi
            x_predicted_error(4,j) = x_predicted_error(4,j) + 2 * pi;
        end
        covariance_predicted  = covariance_predicted + w(j) * x_predicted_error(:,j) * x_predicted_error(:,j)';
    end
    covariance_predicted = covariance_predicted + Q;
    
    %      H = [1 0 0 0 0;0 1 0 0 0];
    %
    %      K = covariance_predicted * H'/(H * covariance_predicted* H' + R);
    %     x_post = x_predict(1:5) + K *(z(:,k) - H * x_predict(1:5));
    %     P = (eye(5) - K * H)* covariance_predicted;
    %     x(1:5,k) = x_post;
    
    %another method
    
    z_prediction = sigmaPointPrediction(1:2,:);
    z_prediction_mean = z_prediction * w';
    % predicted measurement error
    S_error = z_prediction - repmat(z_prediction_mean,[1,2*d+1]);
    S_error_covariance = zeros(2);
    for j = 1:2*d+1
        S_error_covariance = S_error_covariance + w(j) * S_error(:,j) * S_error(:,j)';
    end
    S_error_covariance = S_error_covariance + R;
    
    %互协方差
    C_error_covariance = zeros(5,2);
    for j = 1:2*d+1
        C_error_covariance = C_error_covariance + w(j) * x_predicted_error(:,j) * S_error(:,j)';
    end
    KK = C_error_covariance / S_error_covariance;
    x_post2 = x_predict + KK * (z(:,k) - z_prediction_mean);
    P = covariance_predicted - KK * S_error_covariance * KK';
    NIS(k) = (z(:,k) - z_prediction_mean)' / (S_error_covariance) * (z(:,k) - z_prediction_mean);
    x(:,k) = x_post2;
end

subplot(4,2,[1,3]);
plot(z(1,:),z(2,:),'o-');
hold on
plot(x(1,:),x(2,:));
plot(x(1,44),x(2,44),'g*')
plot(x(1,73),x(2,73),'r*')
plot(z(1,73),z(2,73),'r*');
%             plot(z(1,220-start),z(2,220-start),'g*');
%             plot(z(1,340-start),z(2,340-start),'y*');
% plot(z(1,600-start),z(2,600-start),'m*');
% plot(z(1,900-start),z(2,900-start),'r*');
% plot(x(1,100),x(2,100),'r.','MarkerSize',16);
%             plot(x(1,220-start),x(2,220-start),'g.','MarkerSize',16);
%             plot(x(1,340-start),x(2,340-start),'y.','MarkerSize',16);
% plot(x(1,600-start),x(2,600-start),'m.','MarkerSize',16);
% plot(x(1,900-start),x(2,900-start),'r.','MarkerSize',16);

%set(gca,'XDir','reverse')%对X方向反转
hold off;
title(str)
subplot(4,2,5);
plot(x(4,:).*180/pi);
hold on;
% plot(100,x(4,100).*start/pi,'r*');
%             plot(220-start,x(4,220-start).*180/pi,'g*');
%             plot(340-start,x(4,340-start).*180/pi,'y*');
% plot(600-start,x(4,600-start).*start/pi,'m*');
% plot(900-start,x(4,900-start).*start/pi,'r*');
title('yaw angle');
hold off;
subplot(4,2,7);
plot(x(5,:));
title('yaw rate');
vx = x(3,:) .* cos(x(4,:));
vy = x(3,:) .* sin(x(4,:));

subplot(4,2,2);hold on;
%             zx = radar(:,7);
%             zy = radar(:,8);

plot(vx);
%             plot(zx);
title('vx');
legend('vx','zx');

subplot(4,2,4);
hold on;
plot(vy);
%             plot(zy);
title('vy');
legend('vy','zy');

subplot(4,2,6);
plot(x(3,:));
hold on;
%             plot(sqrt(zx.*zx + zy.* zy));
hold off;
title('v');
legend('vpredict','vradar');

subplot(4,2,8);
plot(NIS);
hold on;
plot(1:NUM_FRAMES,8*ones(1,NUM_FRAMES));
legend('NIS','95%');
title('NIS');
saveas(gcf, ['.\AdditiveS\' str '.png'], 'png');
str = ['sigmaA=' num2str(sigmaA) ' sigmaFai=' num2str(sigmaFai) ' R0=' num2str(sigmaR) ' R4=' num2str(sigmaR)];
%disp([str ' is end']);