clear all;

%% Load landmarks and calculate mean landmarks
landmarks = [];
cnt = 1;
for i = 0:150
    if (i == 103)
        continue; 
    end
    filename = sprintf('face_data/landmark_87/face%03d_87pt.dat', i);
    lm = dlmread(filename,' ', 1, 1);
    landmarks(cnt,:) = [reshape(lm(:,1), 1, size(lm, 1)) reshape(lm(:,2), 1, size(lm, 1))];
    cnt = cnt + 1;
end
landmarks_train = landmarks(1 : 150, :);
mean_lm = mean(landmarks_train);
landmarks_train_minus_mean = landmarks_train - mean_lm;

%% Load images and calculate mean images
data = [];
cnt = 1;
for i = 0:150
    if i == 103
        continue; 
    end
    filename = sprintf('face_data/face/face%03d.bmp', i);
    I = imread(filename);
    I = warpImage_new(I, [reshape(landmarks(cnt,1:87),87,1) reshape(landmarks(cnt,88:end),87,1)],[reshape(mean_lm(1,1:87),87,1) reshape(mean_lm(1,88:end),87,1)]);
    data(cnt, :) = reshape(I, 1, size(I, 1) * size(I, 2));
    cnt = cnt + 1;
end  
X_train = data(1 : 150, :); % wrapped 
mean_x = mean(X_train); % mean_x is a row vector   
X_train_minus_mean = X_train - mean_x;

%% Calculate eigenfaces and eigenvectors
tmp = X_train_minus_mean * X_train_minus_mean';
[U_tmp, D] = svd(tmp);
eigenfaces = [];
for i = 1:150 
    eigenfaces(i,:) = X_train_minus_mean' * U_tmp(:,i);
    eigenfaces(i,:) = eigenfaces(i,:) / norm(eigenfaces(i,:));
end

tmp_lm = landmarks_train_minus_mean * landmarks_train_minus_mean';
[U_tmp_lm, D_lm] = eig(tmp_lm);
[D_lm, I] = sort(diag(D_lm), 'descend');
U_tmp_lm = U_tmp_lm(:, I);

eigenvectors = [];
for i = 1:150 
    eigenvectors(i,:) = landmarks_train_minus_mean' * U_tmp_lm(:,i);
    eigenvectors(i,:) = eigenvectors(i,:) / norm(eigenvectors(i,:));
end

%% random 20 faces
ran_app = zeros(20,256 * 256);
ran_lm = zeros(20,87 * 2);
for i = 1:20
    for j = 1:10
        ran_app_e = normrnd(0.0, 1.0) * sqrt(D(j) / 150.0);
        ran_lm_e = normrnd(0.0, 1.0) * sqrt(D_lm(j) / 150.0);
        ran_app(i, :) = ran_app(i, :) + ran_app_e * eigenfaces(j,:);
        ran_lm(i, :) = ran_lm(i, :) + ran_lm_e * eigenvectors(j,:);
    end
    ran_app(i, :) = ran_app(i, :) + mean_x;
    ran_lm(i, :) = ran_lm(i, :) + mean_lm;
end
figure(1)
for i = 1:20
    I = ran_app(i,:);
    I = warpImage_new(reshape(I,256,256), [reshape(mean_lm(1,1:87),87,1) reshape(mean_lm(1,88:end),87,1)],[reshape(ran_lm(i,1:87),87,1) reshape(ran_lm(i,88:end),87,1)]);
    subplot(4,5,i);
    imshow(uint8(I));
end
print('1_4_random_faces','-dpng');