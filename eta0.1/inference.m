function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

% TODO: FORWARD PROPAGATION CODE
inputI = input;
for i = 1:num_layers
    inputI = model.layers(i).fwd_fn(inputI,model.layers(i).params,...
        model.layers(i).hyper_params,false,[]);
    activations(i) = {inputI};
end

output = activations{end};
