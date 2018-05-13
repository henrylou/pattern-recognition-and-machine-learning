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

male_data = [];
cnt = 0;
for i = 0 : 88
    if (i == 57) continue; end
    filename = sprintf('face_data/male_face/face%03d.bmp', i);
    A = imread(filename);
    cnt = cnt + 1;
    male_data(cnt, :) = reshape(A, 1, size(A, 1) * size(A, 2));
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

male_test = male_data(1:10, :); 
female_test = female_data(1:10, :);
male_data = male_data(11:end, :);
female_data = female_data(11:end, :);

%% Calculate fisher directly
mean_male = mean(male_data);
mean_female = mean(female_data);

C = [male_data; female_data]';
[V,D] = svd(C' * C);
A = [];
for i = 1:size(C,2) 
    tmp = C * V(:,i);
    A(:,i) = D(i,i) * tmp / (norm(tmp)^2);
end
y = A' * (mean_male - mean_female)';
w = C * (inv(D^2 * V') * y);
w = w ./ norm(w);

%% Plot fisher results
boundary_value = (w' * mean_male' + w' * mean_female') / 2.0;
figure(1);
for i = 1 : size(male_data, 1)
    male_point(i) = w' * male_data(i, :)';
end
plot(male_point, 'b+');
hold on;
for i = 1 : size(female_data, 1)
    female_point(i) = w' * female_data(i, :)';
end
plot(female_point, 'r+');
hold on;
for i = 1 : size(male_test, 1)
    male_test_point(i) = w' * male_test(i, :)';
end
plot(male_test_point, 'bo');
hold on;
for i = 1 : size(female_test, 1)
    female_test_point(i) = w' * female_test(i, :)';
end
plot(female_test_point, 'ro');
hold on;
plot(ones(1, 80) * (boundary_value), 'yellow');
legend('male training', 'female training', 'male test', 'female test', 'boundary');    
hold off;
xlabel('Index');
ylabel('Projection');
print('2_1_fisher','-dpng');

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
%% Number of PCA = 10 
k = 10;
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

pca_male_test = [];
for i = 1:size(male_test, 1)
    pca_male_test(i,:) = (male_test(i, :) - mean_data) * eigenfaces(1:k,:)';
end
pca_female_test = [];
for i = 1:size(female_test, 1)
    pca_female_test(i,:) = (female_test(i, :) - mean_data) * eigenfaces(1:k,:)';
end

%% Plot results k = 10
boundary_value = (w' * mean(pca_male)' + w' * mean(pca_female)') / 2.0;
    
figure(2);
for i = 1 : size(pca_male, 1)
    male_point(i) = w' * pca_male(i, :)';
end
plot(male_point, 'b+');
hold on;
for i = 1 : size(pca_female, 1)
    female_point(i) = w' * pca_female(i, :)';
end
plot(female_point, 'r+');
hold on;
for i = 1 : size(pca_male_test, 1)
    male_test_point(i) = w' * pca_male_test(i, :)';
end
plot(male_test_point, 'bo');
hold on;
for i = 1 : size(pca_female_test, 1)
    female_test_point(i) = w' * pca_female_test(i, :)';
end
plot(female_test_point, 'ro');
hold on;
plot(ones(1, 80) * (boundary_value), 'yellow');
legend('male training', 'female training', 'male test', 'female test', 'boundary');
hold off;
xlabel('Index');
ylabel('Projection');
print('2_1_pca_10','-dpng');

%% Number of PCA = 50 
k = 50;
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

pca_male_test = [];
for i = 1:size(male_test, 1)
    pca_male_test(i,:) = (male_test(i, :) - mean_data) * eigenfaces(1:k,:)';
end
pca_female_test = [];
for i = 1:size(female_test, 1)
    pca_female_test(i,:) = (female_test(i, :) - mean_data) * eigenfaces(1:k,:)';
end

%% Plot results k = 50
boundary_value = (w' * mean(pca_male)' + w' * mean(pca_female)') / 2.0;
    
figure(3);
for i = 1 : size(pca_male, 1)
    male_point(i) = w' * pca_male(i, :)';
end
plot(male_point, 'b+');
hold on;
for i = 1 : size(pca_female, 1)
    female_point(i) = w' * pca_female(i, :)';
end
plot(female_point, 'r+');
hold on;
for i = 1 : size(pca_male_test, 1)
    male_test_point(i) = w' * pca_male_test(i, :)';
end
plot(male_test_point, 'bo');
hold on;
for i = 1 : size(pca_female_test, 1)
    female_test_point(i) = w' * pca_female_test(i, :)';
end
plot(female_test_point, 'ro');
hold on;
plot(ones(1, 80) * (boundary_value), 'yellow');
legend('male training', 'female training', 'male test', 'female test', 'boundary');
hold off;
xlabel('Index');
ylabel('Projection');
print('2_1_pca_50','-dpng');