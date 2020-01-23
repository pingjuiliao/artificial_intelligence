function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
;
%disp("Useful Information") ;
%disp("This NN is ");
%disp(input_layer_size) ;
%disp(" X ") ;
%disp(hidden_layer_size) ;
%disp(" X ") ;
%disp(num_labels) ;

Y  = zeros(m, num_labels);
for c = 1:num_labels 
    Y(:, c) = (y == c) ;
end
%%%%%%%%%%%%%%%%%%
%% Cost
%%%%%%  
A1 = [ones(m, 1) X];
A2 = [ones(m, 1) sigmoid(A1 * Theta1')];
hX = sigmoid(A2 * Theta2');
% Pure Cost
J = sum(sum( ((-Y) .* log(hX)) - ((1-Y) .* log(1-hX)) )) ./ m ;
% Regularization Term
J = J + (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))) .* lambda ./ 2 ./ m ;

%disp("Cost") ;
%disp("hX == ");
%disp(size(hX)) ;
%disp("Cost_end") ;
%disp("");
%disp("");

%%%%%%%%%%%%%%%%%%
%% Gradient / Back-Propagation 
%%%%%%  
num_layers     = 3 ;
max_layer_size =  max(hidden_layer_size, num_labels);
% D = zeros(max_layer_size, num_layers) ;

% Dn are column vectors ;
D2 = zeros(num_labels       , hidden_layer_size + 1 ) ;
D1 = zeros(hidden_layer_size, input_layer_size  + 1 ) ;

%disp("Theta2_grad == ");
%disp(size(Theta2_grad));
%disp("Theta1_grad == ");
%disp(size(Theta1_grad));


for t = 1:m 
    a1 = [1, X(t, :)];           % 1 * (l1 + 1)
    z2 = a1 * Theta1' ;     % 1 *    l2      == [ 1 l1+1 ] * [ l1+1  l2 ]
    a2 = [1, sigmoid(z2)];       % 1 * (l2 + 1)
    z3 = a2 * Theta2' ;     % 1 *    l3      == [ 1 num_labels ]
    hX = sigmoid(z3) ;           % 1 *    l3

    % delta_3 == l3 * 1 == num_labels * 1
    delta_3 = [hX - Y(t, :)](:);                                  
    
    % delta_2 == l2 * 1 == [l2+1-1 * l3] * [l3 * 1] .* [l2 * 1]
    delta_2 = (Theta2(:, 2:end)'*delta_3) .* sigmoidGradient(z2')  ;

    % D2 == [ l3 * l2+1 ]  == [ l3 * l2+1 ] + [ l3 * 1 ] * [ 1 * l2+1 ] 
    D2 = D2 + delta_3 * a2 ;

    % D1 == [ l2 * l1+1 ]  == [l2 * 1] * [1 * l1]
    D1 = D1 + delta_2 * a1 ;
end
D1(:, 1)     =   D1(:, 1) ./ m ;
D1(:, 2:end) = ( D1(:, 2:end) .+  lambda .* Theta1(:, 2:end)  ) ./ m ;

D2(:, 1)     =   D2(:, 1) ./ m ;
D2(:, 2:end) = ( D2(:, 2:end) .+  lambda .* Theta2(:, 2:end)  ) ./ m ;





Theta1_grad = D1 ;
Theta2_grad = D2 ;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
