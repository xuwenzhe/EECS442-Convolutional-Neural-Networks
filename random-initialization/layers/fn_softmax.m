% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);

% TODO: FORWARD CODE
output = exp(input);
total = sum(output);
output = output./total;

dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
	% TODO: BACKPROP CODE
    dv_input = zeros(size(input));
    for j = 1:batch_size
        for i = 1:num_classes
            for k = 1:num_classes
                if k == i
                    dv_input(i,j) = dv_input(i,j) + dv_output(i,j)*output(i,j)*(1-output(i,j));
                else
                    dv_input(i,j) = dv_input(i,j) - dv_output(k,j)*output(i,j)*output(k,j);
                end
            end
        end
    end
end
