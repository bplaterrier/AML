function [] = plotRVR(x, y, y_rvm, t, data, MODEL, COLOR)

    figure;
    if data.D==1,
        plot(x, y,'-', 'Color', COLOR.sinc);
        hold on
        plot(x, t, '.', 'Color', COLOR.data);
        plot(x, y_rvm,'r-','LineWidth', 1, 'Color', COLOR.pred);
        plot(x(MODEL.RVs_idx), t(MODEL.RVs_idx),'o', 'Color', COLOR.rv);
        
        xlabel('X');
        ylabel('Y');
        legend('Actual Model', 'Datapoints', 'Regression', 'Relevance Vectors', 'Location', 'NorthWest')
        title('RVR regression on the data');
        
    else
        mesh(gx, gy, reshape(y_rvm,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor', COLOR.sinc);
        mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor' ,COLOR.data);
    end
    
    hold off
    
end

