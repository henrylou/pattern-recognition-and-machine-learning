function adaboost()

% flags
flag_data_subset = 0;
flag_extract_features = 0;
flag_parpool = 0;
flag_boosting = 0;
flag_hard_neg_mining = 0;
flag_detect = 0;
flag_realboosting = 1;

% parpool
if flag_parpool
    delete(gcp('nocreate'));
    parpool(4);
end

% unit tests
test_sum_rect();
test_filters();

% constants
if flag_data_subset
    N_pos = 100;
    N_neg = 100;
else
    % N_pos = 2000;
    % N_neg = 2000;
    
    N_pos = 11838;
    N_neg = 25356;
end
N = N_pos + N_neg;
w = 16;
h = 16;

% (1) haar filter

% load images
if flag_extract_features
    tic;
    I = zeros(N, h, w);
    for i=1:N_pos
        I(i,:,:) = rgb2gray(imread(sprintf('newface16/face16_%06d.bmp',i), 'bmp'));
    end
    for i=1:N_neg
        I(N_pos+i,:,:) = rgb2gray(imread(sprintf('nonface16/nonface16_%06d.bmp',i), 'bmp'));
    end
    fprintf('Loading images took %.2f secs.\n', toc);
end

% construct filters
A = filters_A();
B = filters_B();
C = filters_C();
D = filters_D();
if flag_data_subset
    filters = [A(1:250,:); B(1:250,:); C(1:250,:); D(1:250,:)];
else
    filters = [A; B; C; D];
end
save
% extract features
if flag_extract_features
    tic;
    I = normalize(I);
    II = integral(I);
    features = compute_features(II, filters);
    % save('filters.mat');
    clear I;
    clear II;
    save('features.mat', '-v7.3', 'features');
    fprintf('Extracting %d features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
else
    if flag_hard_neg_mining == 1
        load('features_new.mat','features');
        N = size(features,2);
        N_neg = N - N_pos;
    else
        load('features.mat','features');
    end
end

% perform boosting
if(flag_boosting == 1)
    fprintf('Running AdaBoost with %d features from %d images.\n', size(filters, 1), N);
    tic;
    %% implement this
    h = []; 
    w = ones(N_pos+N_neg,1)/(N_pos+N_neg);
    alpha = [];
    strong_error = [];
    weak_error = [];
    F = [];
    for i = 1:101
        [h,w,alpha,strong_error,weak_error,F] = compute_h(h,w,alpha,strong_error,weak_error,F,features,N_pos,N_neg);
        disp(['Iteration ' num2str(i) ' done']);
    end
    index = h(:,1);
    save('adaboost.mat','h','w','alpha','index','strong_error','weak_error','F');
    %save('adaboost.mat','-v7.3','alpha','index','theta','s','y_hat','eps','err','weak_err');
    fprintf('Running AdaBoost %d with features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
else
    if flag_realboosting == 0
        load('adaboost.mat','h','w','alpha','index','strong_error','weak_error','F');
        %load('adaboost.mat','alpha','index','theta','s','y_hat','eps','err','weak_err');
    end
end

if (flag_realboosting == 1)
    F = size(filters, 1);
    T = 100;
    y(1:N_pos) = ones(N_pos, 1);
    y(N_pos+1:N) = -1 * ones(N_neg, 1);
    weight = ones(N, 1) / N;
    weak_err = zeros(T, F);
    index = zeros(T, 1);   
    F_x = zeros(T, N);
    num_bin = 100;
    p = zeros(T, F, num_bin);
    q = zeros(T, F, num_bin);
    bin_id = zeros(F, N);
    eps = 1e-7;

    % RealBoost code:
    for t=1:T
        for f=1:F
            [bin_id(f,:),~] = discretize(features(f,:), num_bin);
            for i=1:N_pos
                p(t,f,bin_id(f,i)) = p(t,f,bin_id(f,i)) + weight(i);
            end
            for i=1+N_pos:N
                q(t,f,bin_id(f,i)) = q(t,f,bin_id(f,i)) + weight(i);
            end
            weak_err(t, f) = sum(sqrt(p(t,f,:).*q(t,f,:)));
        end
        [~, index(t)] = min(weak_err(t, :));
        h_cur = reshape(0.5 * log((p(t,index(t),:) + eps) ./ (q(t,index(t),:) + eps)), 1, num_bin);
        weight = weight .* exp(-1 * y .* h_cur(bin_id(index(t),:)))';
        if t == 1
            F_x(t, :) = h_cur(bin_id(index(t),:));
        else
            F_x(t, :) = F_x(t - 1, :) + h_cur(bin_id(index(t),:));
        end
        weight = weight / sum(weight);
        disp(['Iteration ' num2str(t) ' Done']);
    end
    save('F_realboost.mat','-v7.3', 'F_x');
end

% (1) top-20 haar filters
%% implement this
%plot_top20_haar(index,filters);
% (2) plot training error
%% implement this
%plot_training_error(strong_error);
% (3) training errors of top-1000 weak classifiers
%% implement this
%plot_weak_classifier_error(weak_error);
% (4) negative positive histograms
%% implement this
%plot_hisgrams(F,N_pos,N_neg);
% (5) plot ROC curves
%% implement this
%plot_ROC(F,N_pos,N_neg)
% (6) detect faces
%% implement this
if flag_detect == 1
    min_scale = 0.075;
    overlap = 0.1;
    score_threshold = 2;
    T = size(h,1);
    for i=1:3
        figure();
        candidates = [];
        if i <= 3
            pic = imread(sprintf('./Testing_Images/Face_%d.jpg', i));
            imshow(pic);
        else
            pic = imread(sprintf('./Testing_Images/Non_Face_%d.jpg', i - 3));
            imshow(pic);
        end

        for scale_factor= 0:0.4:4 % 0:0.4:4
            boxes = [];
            if i > 3
                hard_neg_boxes = [];
            end
            scale = min_scale * 2 ^ scale_factor;
            % scale = min_scale * scale_factor;
            scaled_face = imresize(pic, scale);
            w_0 = size(scaled_face,2) - 16;
            h_0 = size(scaled_face,1) - 16;
            N_img = w_0 * h_0;
            I_crop = zeros(N_img, 16, 16);

            for corner_x=1:w_0
                for corner_y=1:h_0
                    id = (corner_x - 1) * h_0 + corner_y;
                    % I_crop(id, :, :) = double(rgb2gray(imcrop(scaled_face, [corner_x corner_y 16-1 16-1])));
                    I_crop(id,:,:) = double(rgb2gray(scaled_face(corner_y:corner_y+15,corner_x:corner_x+15,:)));
                end
            end
            I_crop = normalize(I_crop);
            II_crop = integral(I_crop);
            %feature_cur = compute_features(II_crop, filters);
            feature_cur = compute_features(II_crop, filters(index, :));
            clear I_crop;
            clear II_crop;
            disp(['Finish extracting features for scale ' num2str(scale)]); 

            for corner_x=1:w_0
                for corner_y=1:h_0
                    % Calculate F(x)
                    h_each = zeros(T, 1);
                    id = (corner_x - 1) * h_0 + corner_y;
                    for t=1:T
                        %h_each(t) = s(t, index(t)) * ((feature_cur(index(t), id) >= theta(t, index(t))) * 2 - 1);
                        %h_each(t) = s(t, index(t)) * ((feature_cur(t, id) >= theta(t, index(t))) * 2 - 1);
                        h_each(t) = h(t,4) * ((feature_cur(t, id) >= h(t,2)) * 2 - 1);
                    end
                    F_x_cur = sum(alpha .* h_each);
                    if (F_x_cur >= score_threshold)
                        box = floor([corner_x corner_y 16-1 16-1] / scale);
                        boxes = [boxes; [box F_x_cur]];
                        if i > 3
                            hard_neg_box = [corner_x corner_y 16-1 16-1];
                            %hard_neg_boxes = [hard_neg_boxes; [hard_neg_box F_x_cur]];
                            hard_neg_boxes = [hard_neg_boxes; hard_neg_box];
                        end
                    end
                end
            end
            candidates = [candidates; boxes];
            if i > 3
                N_hard_neg = size(hard_neg_boxes, 1);
                I_crop = zeros(N_hard_neg, 16, 16);
                for j=1:N_hard_neg
                    % I_crop(j, :, :) = double(rgb2gray(imcrop(scaled_face, hard_neg_boxes(j, 1:4))));
                    I_crop(j, :, :) = double(rgb2gray(scaled_face(corner_y:corner_y+15,corner_x:corner_x+15,:)));
                end
                I_crop = normalize(I_crop);
                II_crop = integral(I_crop);
                features = [features compute_features(II_crop, filters)];
                clear I_crop;
                clear II_crop;
                fprintf('There are now %d negative examples (including %d hard negative ones)', N_neg + N_hard_neg, N_hard_neg);
            end
        end

        top = nms(candidates, overlap);
        for j=1:size(top, 1)
            rectangle('Position', candidates(top(j), 1:4), 'EdgeColor', 'r', 'LineWidth', 1);
        end

        if i <= 3
            print(gcf, '-dpng', sprintf('./pictures/detection_face_%d.png', i));
        else
            print(gcf, '-dpng', sprintf('./pictures/detection_nonface_%d.png', i - 3));
        end
        close all
    end
    save('features_new.mat', '-v7.3', 'features');
end

%% filters
function features = compute_sub_features(II, filters, index)
features = zeros(size(index, 1), size(II, 1));
for j = 1:size(index, 1)
    [rects1, rects2] = filters{index(j),:};
    features(j,:) = apply_filter(II, rects1, rects2);
end
end

function features = compute_features(II, filters)
features = zeros(size(filters, 1), size(II, 1));
for j = 1:size(filters, 1)
    [rects1, rects2] = filters{j,:};
    features(j,:) = apply_filter(II, rects1, rects2);
end
end

function I = normalize(I)
[N,~,~] = size(I);
for i = 1:N;
    image = I(i,:,:);
    sigma = std(image(:));
    I(i,:,:) = I(i,:,:) / sigma;
end
end

function II = integral(I)
[N,H,W] = size(I);
II = zeros(N,H+1,W+1);
for i = 1:N
    image = squeeze(I(i,:,:));
    II(i,2:H+1,2:W+1) = cumsum(cumsum(double(image), 1), 2);
end
end

function sum = apply_filter(II, rects1, rects2)
sum = 0;
% white rects
for k = 1:size(rects1,1)
    r1 = rects1(k,:);
    w = r1(3);
    h = r1(4);
    sum = sum + sum_rect(II, [0, 0], r1) / (w * h * 255);
end
% black rects
for k = 1:size(rects2,1)
    r2 = rects2(k,:);
    w = r2(3);
    h = r2(4);
    sum = sum - sum_rect(II, [0, 0], r2) / (w * h * 255);
end
end

function result = sum_rect(II, offset, rect)
x_off = offset(1);
y_off = offset(2);

x = rect(1);
y = rect(2);
w = rect(3);
h = rect(4);

a1 = II(:, y_off + y + h, x_off + x + w);
a2 = II(:, y_off + y + h, x_off + x);
a3 = II(:, y_off + y,     x_off + x + w);
a4 = II(:, y_off + y,     x_off + x);

result = a1 - a2 - a3 + a4;
end

function rects = filters_A()
count = 1;
w_min = 4;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:2:w_max
    for h = h_min:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/2;
                r1_h = h;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x + r1_w;
                r2_y = r1_y;
                r2_w = w/2;
                r2_h = h;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                rects{count, 1} = r1; % white
                rects{count, 2} = r2; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_B()
count = 1;
w_min = 4;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:w_max
    for h = h_min:2:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w;
                r1_h = h/2;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x;
                r2_y = r1_y + r1_h;
                r2_w = w;
                r2_h = h/2;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                rects{count, 1} = r2; % white
                rects{count, 2} = r1; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_C()
count = 1;
w_min = 6;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:3:w_max
    for h = h_min:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/3;
                r1_h = h;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x + r1_w;
                r2_y = r1_y;
                r2_w = w/3;
                r2_h = h;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                r3_x = r1_x + r1_w + r2_w;
                r3_y = r1_y;
                r3_w = w/3;
                r3_h = h;
                r3 = [r3_x, r3_y, r3_w, r3_h];
                
                rects{count, 1} = [r1; r3]; % white
                rects{count, 2} = r2; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_D()
count = 1;
w_min = 6;
h_min = 6;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:2:w_max
    for h = h_min:2:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/2;
                r1_h = h/2;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x+r1_w;
                r2_y = r1_y;
                r2_w = w/2;
                r2_h = h/2;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                r3_x = x;
                r3_y = r1_y+r1_h;
                r3_w = w/2;
                r3_h = h/2;
                r3 = [r3_x, r3_y, r3_w, r3_h];
                
                r4_x = r1_x+r1_w;
                r4_y = r1_y+r2_h;
                r4_w = w/2;
                r4_h = h/2;
                r4 = [r4_x, r4_y, r4_w, r4_h];
                
                rects{count, 1} = [r2; r3]; % white
                rects{count, 2} = [r1; r4]; % black
                count = count + 1;
            end
        end
    end
end
end

function test_sum_rect()
% 1
I = zeros(1,16,16);
I(1,2:4,2:4) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [2, 2, 3, 3]) == 9);
assert(sum_rect(II, [0, 0], [10, 10, 2, 2]) == 0);

% 2
I = zeros(1,16,16);
I(1,10:16,10:16) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [10, 10, 2, 2]) == 4);

% 3
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 1;
I(1,3:6,11:14) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [11, 3, 6, 6]) == 16);

% 4
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 1;
I(1,3:6,11:14) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [3, 4, 4, 4]) == 12);
assert(sum_rect(II, [0, 0], [7, 4, 4, 4]) == 0);
assert(sum_rect(II, [0, 0], [11, 4, 4, 4]) == 12);
assert(sum_rect(II, [0, 0], [3, 3, 4, 4]) == 16);
assert(sum_rect(II, [0, 0], [11, 3, 4, 4]) == 16);

end

function test_filters()

% A
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,5:8,5:8) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_A();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*2))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [r1s, r2s];
    end
end
assert(max_sum == 1);
assert(max_size == 4*4*2);
assert(isequal(min_f, [1 5 4 4 5 5 4 4]));

% B
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,2:5,2:5) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_B();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*2))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [r1s, r2s];
    end
end
assert(max_sum == 1);
assert(max_size == 4*4*2);
assert(isequal(min_f, [2 6 4 4 2 2 4 4]));

% C
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 255;
I(1,3:6,11:14) = 255;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_C();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r1s(2,3) * r1s(2,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*3))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [reshape(r1s', [1,8]), r2s];
    end
end
assert(max_sum == 2);
assert(max_size == 4*4*3);
assert(isequal(min_f, [3 3 4 4 11 3 4 4 7 3 4 4]));

% D
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,2:5,2:5) = 0;
I(1,6:9,6:9) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_D();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r1s(2,3) * r1s(2,4) + r2s(1,3) * r2s(1,4) + r2s(2,3) * r2s(2,4);
    if(and(f_sum > max_sum, f_size == 4*4*4))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [reshape(r1s', [1,8]), reshape(r2s', [1,8])];
    end
end
assert(max_sum == 2);
assert(max_size == 4*4*4);
assert(isequal(min_f, [6 2 4 4 2 6 4 4 2 2 4 4 6 6 4 4]));
end
end