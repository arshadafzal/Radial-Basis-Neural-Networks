function y = rbf(x,c,theta)

%RBF Gaussian radial basis function
%
%   y = rbf(x, c, theta)
%
%   Inputs:
%       x     : Input vector
%       c     : Center of the radial basis function
%       theta : Shape parameter
%
%   Output:
%       y     : RBF response
         
  y = exp(- theta .^2 * sum((x - c).^2));

end

