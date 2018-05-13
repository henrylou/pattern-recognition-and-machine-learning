function detection()
min_scale = 0.1;
overlap = 0.1;
score_threshold = 2;

for i=1:1%1:6
    figure();
    candidates = [];
    if i <= 3
        pic = imread(sprintf('./Testing_Images/Face_%d.jpg', i));
        imshow(pic);
    else
        pic = imread(sprintf('./Testing_Images/Non_Face_%d.jpg', i - 3));
        imshow(pic);
    end
    
    for scale_factor=1:1%1:0.5:6
        boxes = [];
        if i > 3
            hard_neg_boxes = [];
        end
        % scale = min_scale * 2 ^ scale_factor;
        scale = min_scale * scale_factor;
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
                I_crop(j, :, :) = double(rgb2gray(imcrop(scaled_face, hard_neg_boxes(j, 1:4))));
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
        print(gcf, '-djpeg', sprintf('./project2_code_and_data/pictures/detection_face_%d.jpg', i));
    else
        print(gcf, '-djpeg', sprintf('./project2_code_and_data/pictures/detection_nonface_%d.jpg', i - 3));
    end
    close all
end
end