% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, padding for further work)
% params.W: filter_height x filter_width x filter_depth x num_filters
% params.b: num_filters x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)

[~,~,num_channels,batch_size] = size(input);
[~,~,filter_depth,num_filters] = size(params.W);
assert(filter_depth == num_channels, 'Filter depth does not match number of input channels');

out_height = size(input,1) - size(params.W,1) + 1;
out_width = size(input,2) - size(params.W,2) + 1;
output = zeros(out_height,out_width,num_filters,batch_size);
% TODO: FORWARD CODE
for i = 1:batch_size
    for j = 1:num_filters
        for k = 1:filter_depth
            output(:,:,j,i) = output(:,:,j,i) + conv2(input(:,:,k,i),rot90(params.W(:,:,k,j),2),'valid');
        end
        output(:,:,j,i) = output(:,:,j,i) + params.b(j);
    end
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
    for i=1:batch_size
        for k=1:filter_depth
            for j=1:num_filters
                dv_input(:,:,k,i) = dv_input(:,:,k,i) + conv2(dv_output(:,:,j,i),params.W(:,:,k,j),'full');
            end
        end
    end
    for j=1:num_filters
        for k=1:filter_depth
            for i=1:batch_size
                grad.W(:,:,k,j) = grad.W(:,:,k,j) + conv2(input(:,:,k,i),rot90(dv_output(:,:,j,i),2),'valid');
            end
        end
        grad.b(j) = grad.b(j) + sum(sum(sum(dv_output(:,:,j,:))));
    end
    grad.W = grad.W/batch_size;
    grad.b = grad.b/batch_size;
end
end
