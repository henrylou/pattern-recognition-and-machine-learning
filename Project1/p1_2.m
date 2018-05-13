clear all;

%% Load Images
data = [];
cnt = 1;
for i = 0:177
    if (i == 103)
        continue; 
    end
    filename = sprintf('face_data/face/face%03d.bmp', i);
    I = imread(filename);
    data(cnt, :) = reshape(I, 1, size(I, 1) * size(I, 2));
    cnt = cnt + 1;
end  
X_train = data(1 : 150, :);
X_test = data(151 : 177, :);

%% Load Landmarks
landmarks = [];
cnt = 1;
for i = 0:177
    if (i == 103)
        continue; 
    end
    filename = sprintf('face_data/landmark_87/face%03d_87pt.dat', i);
    lm = dlmread(filename,' ', 1, 1);
    landmarks(cnt,:) = [reshape(lm(:,1), 1, size(lm, 1)) reshape(lm(:,2), 1, size(lm, 1))];
    cnt = cnt + 1;
end
landmarks_train = landmarks(1 : 150, :);
landmarks_test = landmarks(151 : 177, :);

%% Calculate mean face and mean landmark
mean_lm = mean(landmarks_train);
mean_x = mean(X_train); % mean_x is a row vector   
mean_face = reshape(mean_x, size(I, 1), size(I, 2));
figure(1);
imshow(uint8(mean_face));
for i = 1:87
    hold on;
    plot(mean_lm(1,i),mean_lm(1,i+87),'r.','markersize',30);
end
print('1_2_mean_landmarks.png','-dpng');

%% Calculate and display eigen warppings
landmarks_train_minus_mean = landmarks_train - mean_lm;
landmarks_test_minus_mean = landmarks_test - mean_lm;

tmp = landmarks_train_minus_mean * landmarks_train_minus_mean';
[U_tmp, D] = svd(tmp);
eigenvectors = [];
figure(2);
imshow(uint8(mean_face));
for i = 1:150 
    eigenvectors(i,:) = landmarks_train_minus_mean' * U_tmp(:,i);
    eigenvectors(i,:) = eigenvectors(i,:) / norm(eigenvectors(i,:));
end
save eigenvectors.mat;
for i = 1:5
    eigen_warpping_x = 20 * eigenvectors(i,1:87) + mean_lm(1:87);
    eigen_warpping_y = 20 * eigenvectors(i,88:end) + mean_lm(88:end);
    hold on;
    plot(eigen_warpping_x,eigen_warpping_y,'.','markersize',30);
end
hold off;
print('1_2_eigenwarppings.png','-dpng');

%% Reconstruct 27 test landmarks
landmarks_rec = landmarks_test_minus_mean * eigenvectors(1:5,:)' * eigenvectors(1:5,:) + mean_lm;
figure(3);
for i = 1:size(landmarks_test,1)
    subplot(5,6,i);
    imshow(uint8(reshape(X_test(i,:), size(I, 1) , size(I, 2))));
    hold on;
    eigen_warpping_x = landmarks_rec(i,1:87);
    eigen_warpping_y = landmarks_rec(i,88:end);
    plot(eigen_warpping_x,eigen_warpping_y,'r.','markersize',15);
end
print('1_2_reconstruct_landmarks.png','-dpng');

%% Plot the total reconstruct error
error = zeros(1,150);
for i = 1:150
    landmarks_rec_new = landmarks_test_minus_mean * eigenvectors(1:i,:)' * eigenvectors(1:i,:) + mean_lm;
    tmp = (landmarks_rec_new - landmarks_test).^2;
    dist = tmp(:,1:87) + tmp(:,88:174);
    error(i) = sum(sum(sqrt(dist))) / 27 / 87;
end
figure(4);
plot(error);
xlabel('Number of eigenvectors K');
ylabel('Reconstruction error');
print('1_2_reconstruction_error.png','-dpng');


