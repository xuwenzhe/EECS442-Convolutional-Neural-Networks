function [updated_model,updated_dweight] = update_weights(model,grad,hyper_params,dweight)

num_layers = length(grad);
eta = hyper_params.learning_rate;
wd = hyper_params.weight_decay;
alpha = hyper_params.momentum;
updated_model = model;
updated_dweight = dweight;


% TODO: Update the weights of each layer in your model based on the calculated gradients
for i=1:num_layers
    if isempty(dweight{i}.W)
        updated_dweight{i}.W = -eta*wd*model.layers(i).params.W - eta*grad{i}.W;
        updated_dweight{i}.b = -eta*wd*model.layers(i).params.b - eta*grad{i}.b;
    else
        updated_dweight{i}.W = -eta*wd*model.layers(i).params.W - eta*grad{i}.W + alpha*dweight{i}.W;
        updated_dweight{i}.b = -eta*wd*model.layers(i).params.b - eta*grad{i}.b + alpha*dweight{i}.b;
    end
    updated_model.layers(i).params.W = model.layers(i).params.W + updated_dweight{i}.W;
    updated_model.layers(i).params.b = model.layers(i).params.b + updated_dweight{i}.b;
end
    
end