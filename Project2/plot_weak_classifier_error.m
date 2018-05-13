function plot_weak_classifier_error(weak_error)
    figure;
    for i = 1:size(weak_error,1)
        plot(weak_error(i,:));
        hold on;
    end
    hold off;
    legend('T = 0','T = 10','T = 50','T = 100');
    print('2_c','-dpng');
end