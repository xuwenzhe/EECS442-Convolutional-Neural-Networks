% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% labels: batch_size x 1
% ----------------------------------------------------------------------

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));

% TODO: CALCULATE LOSS
batch_size = size(input,2);
indices = sub2ind(size(input),labels,(1:batch_size)');
loss = -(1/batch_size)*sum(log(input(indices)));

dv_input = zeros(size(input));
if backprop
	% TODO: BACKPROP CODE
    dv_input(indices) = -1./input(indices);
end
