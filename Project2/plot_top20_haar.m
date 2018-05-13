function plot_top20_haar(index,filters)
figure;
for i = 1:20
    [rec1, rec2] = filters{index(i),:};
    subplot(4,5,i);
    grey_background = 0.5*ones(16,16);
    image = grey_background;
    
    if size(rec1,1) == 1
        x1 = rec1(1);
        y1 = rec1(2);
        w1 = rec1(3);
        h1 = rec1(4);
        image(x1:x1 + w1 - 1,y1:y1 + h1 - 1) = 0;
    end
    if size(rec1,1) == 2
        x1 = rec1(1,1);
        y1 = rec1(1,2);
        w1 = rec1(1,3);
        h1 = rec1(1,4);
        image(x1:x1 + w1 - 1,y1:y1 + h1 - 1) = 0;
        x2 = rec1(2,1);
        y2 = rec1(2,2);
        w2 = rec1(2,3);
        h2 = rec1(2,4);
        image(x2:x2 + w2 - 1,y2:y2 + h2 - 1) = 0;
    end
    if size(rec2,1) == 1
        x1 = rec2(1);
        y1 = rec2(2);
        w1 = rec2(3);
        h1 = rec2(4);
        image(x1:x1 + w1 - 1,y1:y1 + h1 - 1) = 1;
    end
    if size(rec2,1) == 2
        x1 = rec2(1,1);
        y1 = rec2(1,2);
        w1 = rec2(1,3);
        h1 = rec2(1,4);
        image(x1:x1 + w1 - 1,y1:y1 + h1 - 1) = 1;
        x2 = rec2(2,1);
        y2 = rec2(2,2);
        w2 = rec2(2,3);
        h2 = rec2(2,4);
        image(x2:x2 + w2 - 1,y2:y2 + h2 - 1) = 1;
    end
    imshow(image);
end
print('2_a','-dpng');
end