function plot_ROC(F,N_pos,N_neg)
    figure;
    for i = 1:size(F,1)
        low = min(F(i,:));
        high = max(F(i,:));
        
        x = [];
        y = [];
        for threshold = linspace(low,high,1000)
            fp = sum(F(i,1+N_pos:end) > threshold);
            tp = sum(F(i,1:N_pos) > threshold);
            x = [x fp/N_neg];
            y = [y tp/N_pos];
        end
        plot(x,y);
        hold on;
    end
    hold off;
    xlabel('FPR');
    ylabel('TPR');
    title('ROC');
    legend('T = 10','T = 50','T = 100');
    % print('2_e','-dpng');
    print('3_g','-dpng');
end