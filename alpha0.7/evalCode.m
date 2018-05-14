% EECS 442 HW5
% Define NN
addpath layers;
    % input dimension is 28x28x1
l = [init_layer('conv',struct('filter_size',5,'filter_depth',1,'num_filters',6)) 
    % 24x24x6
	init_layer('pool',struct('filter_size',2,'stride',2)) 
    % 12x12x6
	init_layer('relu',[])
    % 12x12x6
    init_layer('conv',struct('filter_size',5,'filter_depth',6,'num_filters',16)) 
    % 8x8x16
	init_layer('pool',struct('filter_size',2,'stride',2)) 
    % 4x4x16
	init_layer('relu',[])
    % 4x4x16
    init_layer('conv',struct('filter_size',4,'filter_depth',16,'num_filters',120)) 
    % 1x120
	init_layer('relu',[])
	init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',120,'num_out',84))
    init_layer('linear', struct('num_in',84,'num_out',10))
	init_layer('softmax',[])];

model = init_model(l,[28 28 1],10,true);

% Load Data
addpath ../data;
if ~exist('MNIST_loaded')
	% Load training data
	train_data = load_MNIST_images('../data/train-images.idx3-ubyte');
	train_data = reshape(train_data,28,28,1,[]);
	train_label = load_MNIST_labels('../data/train-labels.idx1-ubyte');
	train_label(train_label == 0) = 10; % Remap 0 to 10

	% Load testing data
	test_data = load_MNIST_images('../data/t10k-images.idx3-ubyte');
	test_data = reshape(test_data,28,28,1,[]);
	test_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte');
	test_label(test_label == 0) = 10; % Remap 0 to 10

	MNIST_loaded = true;
end

% Train and Test
input = struct('train', train_data, 'test', test_data);
label = struct('train', train_label, 'test', test_label);
params = struct('learning_rate', 0.1, 'weight_decay', 0.0005,'momentum',0.7,...
    'batch_size', 60, 'plot_interval', 100, 'save_file', 'model.mat');

numIters = 1; %Epochs
 
tic;
[trained_model, loss] = train(model, input, label,params, numIters);
toc
save('loss.mat', 'loss'); %Save the losses 
