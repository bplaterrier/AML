function [] = plotSVR( x, y, y_svm, t, data, MODEL, COLOR )
    
    figure;
    if data.D==1,
        plot(x, y,'-', 'Color', COLOR.sinc);
        hold on
        plot(x, t, '.', 'Color', COLOR.data);
        plot(x, y_svm,'-','LineWidth', 1, 'Color', COLOR.pred);
        plot(x(MODEL.sv_indices), t(MODEL.sv_indices),'o', 'Color', COLOR.rv);
    
        xlabel('X');
        ylabel('Y');  
        legend('Actual Model', 'Datapoints', 'Regression', 'Support Vectors', 'Location', 'NorthWest')
        title('SVR on the data');
        
    else
        mesh(gx, gy, reshape(y_svm,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor', COLOR.sinc);
        plot3(x(MODEL.sv_indices,1), x(MODEL.sv_indices,2), t(MODEL.sv_indices), 'Color' ,COLOR.rv);
    end
    drawnow
    
end

