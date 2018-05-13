function plot_training_error(strong_error)
    figure;
    plot(strong_error);
    xlabel('Iteration T');
    ylabel('Training error of strong classifier');
    print('2_b','-dpng');
end