function classes = proj_2_class(train_file, data_file, lambda)
% Linear Classification
% train_file is the file containing the training data for the model to fit
% data_file is the file containing data to be fit. It can be testing data or any other data
% to be for the model to be applied to.

% The last column of train_file is assumed to be classes. Previous columns
% are assumed to be inputs
% data_file may or may not contain classes. The first n columns are taken
% to be the inputs where n is the number of inputs in train_file.

% lambda is the regularization coefficient.

% import the data
train = readmatrix(train_file);
test = readmatrix(data_file);
train_in = train(:,1:end-1);
train_out = train(:,end);
test_in = test(:,1:length(train_in(1,:)));

% encoding the classes
train_ce = zeros(length(train_out),max(train_out + 1));
for i = 1: length(train_out)
    train_ce(i,train_out(i) + 1) = 1;
end

% calculate the weights
x_train = [ones(length(train_in(:,1)), 1),train_in];
w = (x_train' * x_train + lambda * eye(length(x_train(1,:)))) \ x_train' * train_ce;

% apply model
x_test = [ones(length(test_in(:,1)),1),test_in];
y = x_test * w;
for j = 1:length(train_out)
    [~,i] = max(y(j,:));
    classes(j) = i - 1;
end
classified = classes';