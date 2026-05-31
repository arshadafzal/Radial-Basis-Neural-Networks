function net = rbnn(x_train, y_train, theta, funtol)
%RBNN Orthogonal Least Squares learning for RBF Neural Network
%
%   [c, lw] = rbnn(x_train, y_train, gamma, funtol)
%
%   This function trains a Radial Basis Function Neural Network using
%   Orthogonal Least Squares (OLS) learning.
%
%   Inputs:
%       x_train : Training input data
%                 Size = [number of samples x number of input variables]
%
%       y_train : Training output/target data
%                 Size = [number of samples x 1]
%
%       gamma   : Shape parameter of the radial basis function
%
%       funtol  : Stopping tolerance for mean-squared error
%
%   Outputs:
%       c       : Selected RBF centers
%
%       lw      : Linear output weights of the RBF neural network
%
%   The algorithm selects RBF centers one by one based on their error
%   reduction contribution. The output weights are calculated using the
%   Moore-Penrose pseudoinverse.

% -------------------------------------------------------------
% Initialization
% -------------------------------------------------------------

q = size(x_train, 1);        % Number of training samples
r = size(x_train, 2);        % Number of input variables/features

x = x_train;                 % Copy of training data used for center selection

p = zeros(q, q);             % Design matrix containing candidate RBF responses
err = zeros(q, 1);           % Error reduction contribution of each candidate

c = zeros(q, r);             % Matrix to store selected RBF centers
phi = zeros(q, q);           % Actual RBF design matrix for selected centers

showiterinfo = true;         % Display MSE information at each iteration


% -------------------------------------------------------------
% Construct the initial RBF design matrix
% -------------------------------------------------------------
% Each column of p represents one candidate RBF neuron.
% Initially, each training point is considered as a possible center.

for i = 1:q
    for j = 1:q
        p(i, j) = rbf(x(j, :), x(i, :), theta);  % RBF Function Call
    end
end

% -------------------------------------------------------------
% Select the first RBF center
% -------------------------------------------------------------
% The first center is selected based on maximum error reduction.

for k  = 1:q
    a = p(:, k);
    
    g = (a' * y_train) / (a' * a);
    
    err(k) = g.^2 * (a'*a) / (y_train' * y_train);
end

[~, j] = max(err);            % Index of best candidate

wj = p(:, j);                 % Selected orthogonal basis vector

p(:, j) = [];                 % Remove selected candidate from p
err(j, :) = [];               % Remove corresponding error value

c(1, :) = x(j, :);            % Store selected center
x(j, :) = [];                 % Remove selected center from candidate list


% -------------------------------------------------------------
% Compute network output weights after first selected neuron
% -------------------------------------------------------------

for i  = 1:q
    phi(i, 1) = rbf(x_train(i,:), c(1, :), theta);
end
% Bias column
bias = ones(q, 1);  
lw = pinv([bias phi(:, 1)]) * y_train;
mse = mean(([bias phi(:, 1)] * lw - y_train).^2);

% Display iteration information
if showiterinfo
  disp(['Epochs ' num2str(1) ': MSE = ' num2str(mse)])
end

% -------------------------------------------------------------
% Plot initial MSE
% -------------------------------------------------------------
figure;
semilogy(1, mse,'Marker', 'o', 'Color', 'b','MarkerSize', 6);
hold on;
xlabel('Epochs'); ylabel('MSE');
grid on;
% -------------------------------------------------------------
% Main OLS center selection loop
% -------------------------------------------------------------
% At each iteration:
%   1. Orthogonalize remaining candidate basis vectors
%   2. Compute error reduction contribution
%   3. Select the best candidate center
%   4. Update RBF design matrix
%   5. Compute output weights
%   6. Check stopping condition

for it = 2:q
    alpha = (wj' * p) / (wj' * wj);
    p = p - wj * alpha;
    
    e = size(p, 2);
    
    % ---------------------------------------------------------
    % Compute error reduction for each remaining candidate
    % ---------------------------------------------------------
    for k =1:e
        a = p(:, k);
        
        g = (a' * y_train) / (a' * a);
        err(k) = g.^2 * (a'*a) / (y_train' * y_train);
    end
    % ---------------------------------------------------------
    % Select the candidate with maximum error reduction
    % ---------------------------------------------------------
    
    [~, j] = max(err);
    wj = p(:, j);
    p (:, j) = [];
    err (j, :) = [];
    c(it, :) = x(j, :);
    x (j, :) = [];

    % ---------------------------------------------------------
    % Update actual RBF design matrix for selected centers
    % ---------------------------------------------------------
    for i  = 1:q
        phi(i, it) = rbf(x_train(i,:), c(it, :), theta);
    end
    lw = pinv([bias phi(:, 1:it)]) * y_train;
    mse = mean(([bias phi(:, 1:it)] * lw - y_train).^2);
    
    % ---------------------------------------------------------
    % Plot MSE convergence
    % ---------------------------------------------------------
    
    semilogy(it, mse, 'Marker', 'o', 'Color', 'b','MarkerSize', 6)

    xlabel('Epochs'); ylabel('MSE');
    grid on;
    drawnow;

    % ---------------------------------------------------------
    % Check stopping criterion
    % ---------------------------------------------------------
    if showiterinfo
      disp(['Epochs ' num2str(it) ': MSE = ' num2str(mse)])
    end
    
    
    if mse <= funtol
       fprintf('\nAlgorithm Stopped:Mean-squared error less than specified tolerance\n')
       % Remove unused preallocated rows from center matrix
       for j  = q:-1:it+1
           c(j, :) = [];
       end
    break
    end
    
    if it == q
       fprintf('\nAlgorithm Stopped: Maximum number of neurons reached\')
    break
    end
end

% Store trained RBF neural network parameters in a structure
net.theta = theta;   % Gaussian RBF shape parameter
net.c = c;           % Selected RBF centers
net.lw = lw;         % Linear output weights, including bias weight

end

