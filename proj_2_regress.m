function y = proj_2_regress(train_file, data_file, n_inputs)
% Linear Regression
% train_file is the file containing the training data for the model to fit
% data_file is the file containing data to be fit. It can be testing data or any other data
% to be for the model to be applied to.

% n_inputs is the number of columns of the files which are to be treated as
% inputs.
% Later columns are assumed to be outputs.

% Import the data
train = readmatrix(train_file);
test = readmatrix(data_file);

% split into inputs and outputs
train_in = train(:,1:n_inputs);
train_out = train(:,n_inputs + 1:end);
test_in = test(:,1:n_inputs);

% calculate the weights
x_train = [ones(length(train_in(:,1)), 1),train_in];
w = (x_train' * x_train) \ x_train' * train_out;

% apply the model
x_test = [ones(length(test_in(:,1)),1),test_in];
y = x_test * w;