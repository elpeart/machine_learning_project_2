% Main for Task 2

% Apply the model to the training and testing data
train_class = proj_2_class('train.csv', 'train.csv', 0);
test_class = proj_2_class('train.csv', 'test.csv', 0);

train = readmatrix('train.csv');
test = readmatrix('test.csv');

% training accuracy
c_train = 0;
for i = 1: length(train_class)
    if train_class(i) == train(i, end)
        c_train = c_train + 1;
    end
end
tra_acc = 100 * c_train / length(train_class);
fprintf('The training accuracy for this model is %2.2f%%\n', tra_acc)

% testing accuracy
c_test = 0;
for i = 1: length(test_class)
    if test_class(i) == test(i, end)
        c_test = c_test + 1;
    end
end
tst_acc = 100 * c_test / length(test_class);
fprintf('The testing accuracy for this model is %2.2f%%\n', tst_acc)