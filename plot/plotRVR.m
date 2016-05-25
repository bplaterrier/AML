function [] = plotRVR(x, y, y_rvm, t, data, MODEL, COLOR)

    figure;
    if data.D==1,
        hold on
        
        % Data
        plot(x, y,'-', 'Color', COLOR.sinc);
       
        % Model
        plot(x, t, '.', 'Color', COLOR.data);
        
        % Regression
        plot(x, y_rvm,'r-','LineWidth', 1, 'Color', COLOR.pred);
            
        % Relevance Vectors
        plot(x(MODEL.RVs_idx), t(MODEL.RVs_idx),'o', 'Color', COLOR.rv);
        
        % Variable Noise tube
        area(x, MODEL.mu_star + MODEL.sigma_star, -1, 'EdgeAlpha', 0, 'FaceColor', 'r', 'FaceAlpha', 0.1);
        area(x, MODEL.mu_star - MODEL.sigma_star, -1, 'EdgeAlpha', 0, 'FaceColor', 'w', 'FaceAlpha', 1);
        
        % Fixed noise tube (1/beta)
        plot(x, y_rvm + sqrt(1/MODEL.beta),'r:','LineWidth', 1, 'Color', COLOR.pred);
        plot(x, y_rvm - sqrt(1/MODEL.beta),'r:','LineWidth', 1, 'Color', COLOR.pred);
        
        % Redraw
        plot(x, y,'-', 'Color', COLOR.sinc);
        plot(x, t, '.', 'Color', COLOR.data);
        plot(x, y_rvm,'r-','LineWidth', 1, 'Color', COLOR.pred);
        plot(x(MODEL.RVs_idx), t(MODEL.RVs_idx),'o', 'Color', COLOR.rv);
        
        % Legends & Axis options
        xlabel('X', 'Interpreter', 'LaTex');
        ylabel('Y', 'Interpreter', 'LaTex');
        legend({'Actual Model', 'Datapoints', 'Regression', 'Relevance Vectors'}, 'Location', 'NorthWest', 'Interpreter', 'LaTex')
        
        title_string = sprintf('RVR + RBF Kernel: $\\sigma$ = %g, RV = %d, $\\epsilon_{est}$= %g', MODEL.lengthScale, length(MODEL.RVs_idx), sqrt(1/MODEL.beta)); 
        title(title_string, 'Interpreter', 'LaTex');
        
    else
        mesh(gx, gy, reshape(y_rvm,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor', COLOR.sinc);
        mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', COLOR.sinc, 'facecolor' ,COLOR.data);
    end
    
    hold off
    
end

