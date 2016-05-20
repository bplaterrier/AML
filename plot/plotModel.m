function [] = plotModel(x, y, t, data, COLOR)    

    figure;
    whitebg(1,'w');
    clf;
    hold on;
    
    if data.D==1,
        plot(x, y,'-', 'Color', COLOR.sinc);
        plot(x, t, '.', 'Color', COLOR.data);
        xlabel('X');
        ylabel('Y');
        legend('Target Function', 'Actual Data');
        title('Model of the data');
    else
        mesh(gx, gy, reshape(y,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor', COLOR.sinc);
        mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor' ,COLOR.data);
    end

    set(gca,'FontSize',12)
    drawnow

end

