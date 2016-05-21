function [] = plotModel(x, y, t, data, COLOR)    

    figure;
    clf;
    hold on;
    
    if data.D==1,
        plot(x, y,'-', 'Color', COLOR.sinc);
        plot(x, t, '.', 'Color', COLOR.data);
        xlabel('X', 'Interpreter', 'LaTex');
        ylabel('Y', 'Interpreter', 'LaTex');
        legend({'Target Function', 'Actual Data'}, 'Interpreter', 'LaTex');
        title('Model of the data', 'Interpreter', 'LaTex');
    else
        mesh(gx, gy, reshape(y,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor', COLOR.sinc);
        mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor' ,COLOR.data);
    end

    set(gca,'FontSize',12)
    drawnow

end

