function yp = simulate(x_test, net)
% SIMULATE Predict output using a trained RBF neural network
%
%   yp = simulate(x_test, net)
%
%   This function computes the predicted output of a trained Radial Basis
%   Function Neural Network for the given test input data.
%
%   Inputs:
%       x_test : Test input data
%                Size = [number of test samples x number of input variables]
%
%       net    : Trained RBF neural network structure
%
%
%   Output:
%       yp     : Predicted output for the test data
%                Size = [number of test samples x 1]
    theta = net.theta;
    c = net.c;
    lw = net.lw;
    
    % -------------------------------------------------------------
    % Determine number of test samples and selected RBF centers
    % -------------------------------------------------------------
    q = size(x_test, 1);
    t = size(c, 1);
    phi_test = zeros(q, t);
    
    % -------------------------------------------------------------
    % Construct RBF design matrix for test data
    % -------------------------------------------------------------
    % Each test sample is evaluated against each selected RBF center.
    for i  = 1:t
        for j  = 1:q
            phi_test(j, i) = rbf(x_test(j, :), c(i, :), theta);
        end
    end
    % Bias column
    bias = ones(q, 1); 
    phi_test = [bias phi_test];
    
    % -------------------------------------------------------------
    % Predict output
    % -------------------------------------------------------------
    
    yp = phi_test * lw;


end

