function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
t=[0 0];

for iter = 1:num_iters
	     ter1=X*theta;
ter2=ter1-y;
temp0=theta(1) -((alpha/m)*(sum(X(:,1)'*ter2)));
temp1=theta(2) -((alpha/m)*(sum(X(:,2)'*ter2)));
theta(1)=temp0;
theta(2)=temp1;

((sum(X(:,1)'*ter2)))
((sum(X(:,2)'*ter2)))


    J_history(iter) = computeCost(X, y, theta);

end

end
