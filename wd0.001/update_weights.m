function updated_model = update_weights(model,grad,hyper_params)

num_layers = length(grad);
eta = hyper_params.learning_rate;
wd = hyper_params.weight_decay;
updated_model = model;

% TODO: Update the weights of each layer in your model based on the calculated gradients
for i=1:num_layers
   updated_model.layers(i).params.W = (1-eta*wd)*model.layers(i).params.W - eta*grad{i}.W;
   updated_model.layers(i).params.b = (1-eta*wd)*model.layers(i).params.b - eta*grad{i}.b;
end
    
end