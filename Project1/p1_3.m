clear all;

%% Load landmarks and calculate mean landmarks
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
mean_lm = mean(landmarks_train);
landmarks_test_minus_mean = landmarks_test - mean_lm;

%% Load images and warp images
data = [];
cnt = 1;
for i = 0:177
    if i == 103
        continue; 
    end
    filename = sprintf('face_data/face/face%03d.bmp', i);
    I = imread(filename);
    if i <= 150
        I = warpImage_new(I, reshape(landmarks(cnt, :),87,2),reshape(mean_lm,87,2));
    end
    data(cnt, :) = reshape(I, 1, size(I, 1) * size(I, 2));
    cnt = cnt + 1;
end  
X_train = data(1 : 150, :); % wrapped
X_test = data(151 : 177, :); % unwrapped
mean_x = mean(X_train); % mean_x is a row vector   
X_train_minus_mean = X_train - mean_x;

%% Calculate and display eigenfaces
tmp = X_train_minus_mean * X_train_minus_mean';
[U_tmp, D] = svd(tmp);
eigenfaces = [];
for i = 1:150 
    eigenfaces(i,:) = X_train_minus_mean' * U_tmp(:,i);
    eigenfaces(i,:) = eigenfaces(i,:) / norm(eigenfaces(i,:));
end

%% Reconstruct landmarks and test images
load('eigenvectors.mat');

landmarks_rec = landmarks_test_minus_mean * eigenvectors(1:10,:)' * eigenvectors(1:10,:) + mean_lm;
X_test_warp = [];
for i = 1:27
    I = X_test(i,:);
    I = warpImage_new(reshape(I,256,256), reshape(landmarks_test(i, :),87,2),reshape(mean_lm,87,2));
    X_test_warp(i,:) = reshape(I,1,size(I,1) * size(I,2));
end
X_rec = (X_test_warp - mean_x) * eigenfaces(1:10,:)' * eigenfaces(1:10,:) + mean_x;
figure(1); % mean position
for i = 1:size(X_rec,1)
    subplot(5,6,i);
    imshow(uint8(reshape(X_rec(i,:), size(I, 1) , size(I, 2))));
end
print('1_3_27_reconstruct_faces_unwarpback','-dpng');

X_rec_unwarpped = []; 
for i = 1:27
    I = X_rec(i,:);
    I = warpImage_new(reshape(I,256,256), reshape(mean_lm,87,2),[reshape(landmarks_rec(i,1:87),87,1) reshape(landmarks_rec(i,88:end),87,1)]);
    X_rec_unwarpped(i,:) = reshape(I,1,size(I,1) * size(I,2));
end
figure(2); % mean position -> reconstructed position of landmarks
for i = 1:size(X_rec,1)
    subplot(5,6,i);
    imshow(uint8(reshape(X_rec_unwarpped(i,:), size(I, 1) , size(I, 2))));
end
print('1_3_27_reconstruct_faces_warpback','-dpng');
%% Plot the total reconstruct error
error = zeros(1,20);
X_rec_unwarpped_new = zeros(27,256 * 256);
for i = 1:150
    X_rec_new = (X_test_warp - mean_x) * eigenfaces(1:i,:)' * eigenfaces(1:i,:) + mean_x;
    for k = 1:27
        I = X_rec_new(k,:);
        I = warpImage_new(reshape(I,256,256), reshape(mean_lm,87,2),[reshape(landmarks_rec(k,1:87),87,1) reshape(landmarks_rec(k,88:end),87,1)]);
        X_rec_unwarpped_new(k,:) = reshape(I,1,size(I,1) * size(I,2));
        fprintf('%d',k);
    end
    error(i) = sum(sum((X_rec_unwarpped_new - X_test).^2)) / size(X_test,1) / size(X_test,2);
    fprintf('\n%d\n',i);
end
figure(3);
plot(error);
xlabel('Number of eigenfaces K');
ylabel('Reconstruction error');
print('1_3_27_reconstruct_error','-dpng');