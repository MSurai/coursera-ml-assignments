function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
disp(sprintf('Current J(theta): %f', computeCost(X, y, theta)));


% My approach
tmp = theta;
for j = 1:length(theta)
    tmp(j) = theta(j) - alpha * 1 / m * (X * theta - y)' * X(:,j);
end
theta = tmp;

% Vectorized approach http://stackoverflow.com/questions/20735406/vectorization-of-a-gradient-descent-code
%theta = theta - alpha * 1 / m * X' * (X * theta - y);



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
