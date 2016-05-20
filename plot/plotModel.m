function [] = plotModel(x, y, t, data, COL_sinc, COL_data)    

    figure(1);
    whitebg(1,'w');
    clf;
    hold on;
    
    if data.D==1,
        plot(x, y,'-', 'Color', COL_sinc);
        plot(x, t, '.', 'Color', COL_data);
    else
        mesh(gx, gy, reshape(y,size(gx)), 'edgecolor', COL_sinc, 'facecolor', COL_sinc);
        mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', COL_sinc, 'facecolor' ,COL_data);
    end

    set(gca,'FontSize',12)
    drawnow
    
    xlabel('X');
    ylabel('Y');



end

