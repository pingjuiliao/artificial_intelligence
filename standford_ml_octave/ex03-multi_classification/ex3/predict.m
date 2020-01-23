function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 = [25 * 401] , Theta2 = [10 * 26]
% A1 = X = [m * n] = [#samples, #features] = [5000 * 400]
% Ans = p = [m * 1]
A1 = [ones(m, 1) X ];             % 5000 * 401
Z2 = A1 * Theta1';                % 5000 * 25
A2 = [ones(m, 1) sigmoid(Z2)];    % 5000 * 26
Z3 = A2 * Theta2';                % 5000 * 10
hX = sigmoid(Z3) ;                % 5000 * 10

[pred, iPred] = max(hX, [], 2);
p = iPred;




% =========================================================================


end
