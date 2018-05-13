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

%% Calculate the mean meanface and update X_train and X_test by substracting
% the mean face
mean_x = mean(X_train); % mean_x is a row vector   
X_train_minus_mean = X_train - mean_x;
X_test_minus_mean = X_test - mean_x;
mean_face = reshape(mean_x, size(I, 1), size(I, 2));
figure(1);
imshow(uint8(mean_face));
print('1_1_train_mean.png','-dpng');

%% Calculate and display eigenfaces
tmp = X_train_minus_mean * X_train_minus_mean';
[U_tmp, D] = svd(tmp);
eigenfaces = [];
figure(2);
for i = 1:150 
    eigenfaces(i,:) = X_train_minus_mean' * U_tmp(:,i);
    if i <= 20
        subplot(4,5,i);
        % imagesc((reshape(eigenfaces(i,:), size(I, 1) , size(I, 2))));???
        face = 255 * mat2gray((reshape(eigenfaces(i,:), size(I, 1) , size(I, 2))));
        imshow(uint8(face));
    end
    eigenfaces(i,:) = eigenfaces(i,:) / norm(eigenfaces(i,:));
end
print('1_1_20_eigenfaces.png','-dpng');

%% Reconstruct 27 test faces
X_rec = X_test_minus_mean * eigenfaces(1:20,:)' * eigenfaces(1:20,:) + mean_x;
figure(3);
for i = 1:size(X_test,1)
    subplot(5,6,i);
    imshow(uint8(reshape(X_rec(i,:), size(I, 1) , size(I, 2))));
end
print('1_1_27_reconstruct_faces.png','-dpng');

%% Plot the total reconstruct error
error = zeros(1,20);
for i = 1:150%20
    X_rec_new = X_test_minus_mean * eigenfaces(1:i,:)' * eigenfaces(1:i,:) + mean_x;
    error(i) = sum(sum((X_rec_new - X_test).^2)) / size(X_test,1) / size(X_test,2);
end
figure(4);
plot(error);
xlabel('Number of eigenvectors K');
ylabel('Reconstruction error');
print('1_1_reconstruction_error.png','-dpng'); 
