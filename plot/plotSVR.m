function [] = plotSVR( x, y, y_svm, t, data, MODEL, OPTIONS, COLOR )
    
    if isempty(COLOR)
        COLOR.sinc = 'k';     % color of the actual function
        COLOR.data = 'b';     % color of the real data
        COLOR.pred = 'r';     % color of the prediction
        COLOR.rv = 'k';       % color of the relevance vectors
    end
    
    figure;
    if data.D==1,
        hold on
        
        % Data
        plot(x, y,'-', 'Color', COLOR.sinc);
        
        % Model
        plot(x, t, '.', 'Color', COLOR.data);
        
        % Regression
        plot(x, y_svm,'-',  'LineWidth', 1, 'Color', COLOR.pred);
        
        % Support Vectors
        plot(x(MODEL.sv_indices), t(MODEL.sv_indices),'o', 'Color', COLOR.rv);
        
        % Epsilon-tube
        area(x, y_svm + OPTIONS.epsilon, -1, 'EdgeAlpha', 0, 'FaceColor', 'r', 'FaceAlpha', 0.1);
        area(x, y_svm - OPTIONS.epsilon, -1, 'EdgeAlpha', 0, 'FaceColor', 'w', 'FaceAlpha', 1);
        
        
        % Data
        plot(x, y,'-', 'Color', COLOR.sinc);
        plot(x, t, '.', 'Color', COLOR.data);
        plot(x, y_svm,'-',  'LineWidth', 1, 'Color', COLOR.pred);
        plot(x(MODEL.sv_indices), t(MODEL.sv_indices),'o', 'Color', COLOR.rv);
        
        % Legends & Axis options
        xlabel('X', 'Interpreter', 'LaTex');
        ylabel('Y', 'Interpreter', 'LaTex');
        legend({'Actual Model', 'Datapoints', 'Regression', 'Support Vectors', '$\epsilon$-tube'}, 'Interpreter', 'LaTex', 'Location', 'NorthWest')
        
        title_string = sprintf('$\\epsilon$-SVR + RBF kernel: $\\epsilon$ = %g, $\\sigma$ = %g, $C$ =%d, $SV$ = %d', OPTIONS.epsilon, OPTIONS.lengthScale, OPTIONS.C, MODEL.totalSV);
        title(title_string, 'Interpreter', 'LaTex');
        
    else
        mesh(gx, gy, reshape(y_svm,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor', COLOR.sinc);
        plot3(x(MODEL.sv_indices,1), x(MODEL.sv_indices,2), t(MODEL.sv_indices), 'Color' ,COLOR.rv);
    end
    drawnow
    
end

