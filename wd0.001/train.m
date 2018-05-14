function [model, loss] = train(model,input,label,params,numIters)

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% Learning rate
if isfield(params,'learning_rate') lr = params.learning_rate;
else lr = .01; end
% Weight decay
if isfield(params,'weight_decay') wd = params.weight_decay;
else wd = .0005; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 128; end

% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model.mat'; end

if isfield(params,'plot_interval') plot_interval = params.plot_interval;
else plot_interval = 20; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd);

num_train = size(input.train,4);
num_batches = int32(num_train/batch_size);
loss = -1.0*ones(2,numIters*num_batches/plot_interval);

% refresh graph with train loss data
x = 1:num_batches;
y1 = -1.0*ones(1,num_batches);
figure,hold
h1 = plot(x,y1,'o-',...
    'Color','r',...
    'MarkerSize',5,...
    'MarkerEdgeColor','red',...
    'MarkerFaceColor',[1 .6 .6]);

y2 = -1.0*ones(1,num_batches);
h2 = plot(x,y2,'s-',...
    'Color','b',...
    'MarkerSize',7,...
    'MarkerEdgeColor','blue',...
    'MarkerFaceColor',[.6 .6 1]);
xlim([0 1000])
ylim([0 3])
xlabel('Iterations');
ylabel('Crossentropy Loss');
legend('Training','Test')


dlmwrite('x_y1_y2',[double(x'),double(y1'),double(y2')],'delimiter','\t','precision',6);
for i = 1:numIters % "epoch"
	% TODO: Training code
    idx = randsample(num_train, num_train, false);
    for j = 1:num_batches % "iteration"
        fprintf('Iteration %i\n',j);
        batch = input.train(:,:,:,idx( (j-1)*batch_size+1 : j*batch_size ));
        batch_label = label.train(idx( (j-1)*batch_size+1 : j*batch_size ));
        
        [output,activations] = inference(model,batch);
        [train_loss,dv_output] = loss_crossentropy(output,batch_label,[],true);
        fprintf('Loss = %d\n', train_loss);
        [grad] = calc_gradient(model,batch,activations,dv_output);
        model = update_weights(model,grad,update_params);
        y1(j) = train_loss;
        set(h1,'XData',x(y1>0))
        set(h1,'YData',y1(y1>0))
        drawnow
        if mod((i-1)*num_batches+j,plot_interval) == 0
            test_output = inference(model,input.test);
            test_loss = loss_crossentropy(test_output,label.test,[],false);
            lossIndex = int32(((i-1)*num_batches+j)/plot_interval);
            loss(1,lossIndex) = train_loss;
            loss(2,lossIndex) = test_loss;
            y2(j) = test_loss;
            set(h2,'XData',x(y2>0))
            set(h2,'YData',y2(y2>0))
            drawnow
            % check whether stop optimizing
            [~,test_predictions] = max(test_output);
            test_predictions = reshape(test_predictions,size(label.test));
            test_accuracy = sum(test_predictions == label.test)/numel(label.test);
            fprintf('Accuracy (test data) = %d\n',test_accuracy);
            % backup model after training
            save(save_file,'model');
            if (test_accuracy > 0.96)
                dlmwrite('x_y1_y2',[double(x'),double(y1'),double(y2')],'delimiter','\t','precision',6);
                return
            end
        end
    end
end
dlmwrite('x_y1_y2',[double(x'),double(y1'),double(y2')],'delimiter','\t','precision',6);
end
