function plot_hisgrams(F,N_pos,N_neg);
    figure;
    histogram(F(1,1:N_pos),'BinWidth',0.3);
    hold on;
    histogram(F(1,N_pos+1:end),'BinWidth',0.3);
    hold off;
    legend('Pos(face)','Neg(non-face)');
    % print('2_d_10','-dpng');
    print('3_h_10','-dpng');
    
    figure;
    histogram(F(2,1:N_pos),'BinWidth',0.3);
    hold on;
    histogram(F(2,N_pos+1:end),'BinWidth',0.3);
    hold off;
    legend('Pos(face)','Neg(non-face)');
    % print('2_d_50','-dpng');
    print('3_h_50','-dpng');
    
    figure;
    histogram(F(3,1:N_pos),'BinWidth',0.5);
    hold on;
    histogram(F(3,N_pos+1:end),'BinWidth',0.5);
    hold off;
    legend('Pos(face)','Neg(non-face)');
    % print('2_d_100','-dpng');
    print('3_h_100','-dpng');
end