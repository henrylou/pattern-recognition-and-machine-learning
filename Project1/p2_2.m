clear all;

%% load gender dataset
cnt = 0;
female_data = [];
for i = 0 : 84
    filename = sprintf('face_data/female_face/face%03d.bmp', i);
    A = imread(filename);
    cnt = cnt + 1;
    female_data(cnt, :) = reshape(A, 1, size(A, 1) * size(A, 2));
end
cnt = 0;
female_lm = [];
for i = 0 : 84
    fileID = fopen(sprintf('face_data/female_landmark_87/face%03d_87pt.txt', i));
    cnt = cnt + 1;
    inverse = fscanf(fileID, '%f', 2 * 87)';
    female_lm(cnt, :) = [inverse([1:87]*2 - 1), inverse([1:87]*2)];
    fclose(fileID);
end

male_data = [];
cnt = 0;
for i = 0 : 88
    if (i == 57) continue; end
    filename = sprintf('face_data/male_face/face%03d.bmp', i);
    A = imread(filename);
    cnt = cnt + 1;
    male_data(cnt, :) = reshape(A, 1, size(A, 1) * size(A, 2));
end
cnt = 0;
male_lm = [];
for i = 0 : 88
    if (i == 57) continue; end
    fileID = fopen(sprintf('face_data/male_landmark_87/face%03d_87pt.txt', i));
    cnt = cnt + 1;
    inverse = fscanf(fileID, '%f', 2 * 87)';
    male_lm(cnt, :) = [inverse([1:87]*2 - 1), inverse([1:87]*2)];
    fclose(fileID);
end

unknown_data = [];
cnt = 0;
for i = 0 : 3
    filename = sprintf('face_data/unknown_face/face%03d.bmp', i);
    A = imread(filename);
    cnt = cnt + 1;
    unknown_data(cnt, :) = reshape(A, 1, size(A, 1) * size(A, 2));
end
cnt = 0;
unknown_lm = [];
for i = 0 : 3
    fileID = fopen(sprintf('face_data/unknown_landmark_87/face%03d_87pt.txt', i));
    cnt = cnt + 1;
    inverse = fscanf(fileID, '%f', 2 * 87)';
    unknown_lm(cnt, :) = [inverse([1:87]*2 - 1), inverse([1:87]*2)];
    fclose(fileID);
end

male_test = male_data(1:10, :); 
female_test = female_data(1:10, :);
male_data = male_data(11:end, :);
female_data = female_data(11:end, :);
mean_male = mean(male_data);
mean_female = mean(female_data);

male_test_lm = male_lm(1:10,:);
female_test_lm = female_lm(1:10, :);
male_lm = male_lm(11:end, :);
female_lm = female_lm(11:end, :);
mean_male_lm = mean(male_lm);
mean_female_lm = mean(female_lm);

%% Applying PCA
data = [male_data; female_data];
mean_data = mean(data);
data_minus_mean = data - mean_data;
tmp = data_minus_mean * data_minus_mean';
[U_tmp, D] = svd(tmp);
eigenfaces = [];
for i = 1:150 
    eigenfaces(i,:) = data_minus_mean' * U_tmp(:,i);
    eigenfaces(i,:) = eigenfaces(i,:) / norm(eigenfaces(i,:));
end

lm = [male_lm; female_lm];
mean_lm = mean(lm);
lm_minus_mean = lm - mean_lm;
tmp_lm = lm_minus_mean * lm_minus_mean';
[U_tmp_lm, D_lm] = svd(tmp_lm);
eigenvectors = [];
for i = 1:150 
    eigenvectors(i,:) = lm_minus_mean' * U_tmp_lm(:,i);
    eigenvectors(i,:) = eigenvectors(i,:) / norm(eigenvectors(i,:));
end

%% Number of PCA = 10 
k = 10;
% faces
pca_male = [];
for i = 1:size(male_data, 1)
    pca_male(i,:) = (male_data(i, :) - mean_data) * eigenfaces(1:k,:)';
end
pca_female = [];
for i = 1:size(female_data, 1)
    pca_female(i,:) = (female_data(i, :) - mean_data) * eigenfaces(1:k,:)';
end    
S = pca_male' * pca_male + pca_female' * pca_female;
w = inv(S) * (mean(pca_male) - mean(pca_female))';
w = w ./ norm(w);
% landmarks
pca_male_lm = [];
for i = 1:size(male_lm, 1)
    pca_male_lm(i,:) = (male_lm(i, :) - mean_lm) * eigenvectors(1:k,:)';
end
pca_female_lm = [];
for i = 1:size(female_data, 1)
    pca_female_lm(i,:) = (female_lm(i, :) - mean_lm) * eigenvectors(1:k,:)';
end    
S_lm = pca_male_lm' * pca_male_lm + pca_female_lm' * pca_female_lm;
w_lm = inv(S_lm) * (mean(pca_male_lm) - mean(pca_female_lm))';
w_lm = w_lm ./ norm(w_lm);
% faces test
pca_male_test = [];
for i = 1:size(male_test, 1)
    pca_male_test(i,:) = (male_test(i, :) - mean_data) * eigenfaces(1:k,:)';
end
pca_female_test = [];
for i = 1:size(female_test, 1)
    pca_female_test(i,:) = (female_test(i, :) - mean_data) * eigenfaces(1:k,:)';
end
% landmarks test
pca_male_test_lm = [];
for i = 1:size(male_test_lm, 1)
    pca_male_test_lm(i,:) = (male_test_lm(i, :) - mean_lm) * eigenvectors(1:k,:)';
end
pca_female_test_lm = [];
for i = 1:size(female_test_lm, 1)
    pca_female_test_lm(i,:) = (female_test_lm(i, :) - mean_lm) * eigenvectors(1:k,:)';
end

figure(1);
for i = 1 : size(male_data)
    hold on;
    plot(w' * pca_male(i, :)', w_lm' * pca_male_lm(i, :)', 'b+');
end
for i = 1 : size(female_data)
    hold on;
    plot(w' * pca_female(i, :)', w_lm' * pca_female_lm(i, :)', 'r+');
end
for i = 1 : 10
    hold on;
    plot(w' * pca_male_test(i, :)', w_lm' * pca_male_test_lm(i, :)', 'bo');
end
for i = 1 : 10
    hold on;
    plot(w' * pca_female_test(i, :)', w_lm' * pca_female_test_lm(i, :)', 'ro');
end
hold off;
% legend('male training', 'female training', 'male test', 'female test');
print('2_2_pca_10','-dpng');

%% Number of PCA = 50 
k = 50;
% faces
pca_male = [];
for i = 1:size(male_data, 1)
    pca_male(i,:) = (male_data(i, :) - mean_data) * eigenfaces(1:k,:)';
end
pca_female = [];
for i = 1:size(female_data, 1)
    pca_female(i,:) = (female_data(i, :) - mean_data) * eigenfaces(1:k,:)';
end    
S = pca_male' * pca_male + pca_female' * pca_female;
w = inv(S) * (mean(pca_male) - mean(pca_female))';
w = w ./ norm(w);
% landmarks
pca_male_lm = [];
for i = 1:size(male_lm, 1)
    pca_male_lm(i,:) = (male_lm(i, :) - mean_lm) * eigenvectors(1:k,:)';
end
pca_female_lm = [];
for i = 1:size(female_data, 1)
    pca_female_lm(i,:) = (female_lm(i, :) - mean_lm) * eigenvectors(1:k,:)';
end    
S_lm = pca_male_lm' * pca_male_lm + pca_female_lm' * pca_female_lm;
w_lm = inv(S_lm) * (mean(pca_male_lm) - mean(pca_female_lm))';
w_lm = w_lm ./ norm(w_lm);
% faces test
pca_male_test = [];
for i = 1:size(male_test, 1)
    pca_male_test(i,:) = (male_test(i, :) - mean_data) * eigenfaces(1:k,:)';
end
pca_female_test = [];
for i = 1:size(female_test, 1)
    pca_female_test(i,:) = (female_test(i, :) - mean_data) * eigenfaces(1:k,:)';
end
% landmarks test
pca_male_test_lm = [];
for i = 1:size(male_test_lm, 1)
    pca_male_test_lm(i,:) = (male_test_lm(i, :) - mean_lm) * eigenvectors(1:k,:)';
end
pca_female_test_lm = [];
for i = 1:size(female_test_lm, 1)
    pca_female_test_lm(i,:) = (female_test_lm(i, :) - mean_lm) * eigenvectors(1:k,:)';
end

figure(2);
for i = 1 : size(male_data)
    hold on;
    plot(w' * pca_male(i, :)', w_lm' * pca_male_lm(i, :)', 'b+');
end
for i = 1 : size(female_data)
    hold on;
    plot(w' * pca_female(i, :)', w_lm' * pca_female_lm(i, :)', 'r+');
end
for i = 1 : 10
    hold on;
    plot(w' * pca_male_test(i, :)', w_lm' * pca_male_test_lm(i, :)', 'bo');
end
for i = 1 : 10
    hold on;
    plot(w' * pca_female_test(i, :)', w_lm' * pca_female_test_lm(i, :)', 'ro');
end
% legend('male training','female training', 'male test', 'female test');
print('2_2_pca_50','-dpng');