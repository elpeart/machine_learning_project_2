% Main for Task 3

for lambda = [0.01, 0.1, 0.5, 1, 5, 10]  % run for each lambda
    
    % Apply the model to the data
    train_class = proj_2_class('zipcode_train.csv', 'zipcode_train.csv',lambda);
    test_class = proj_2_class('zipcode_train.csv', 'zipcode_test.csv',lambda);
    
    train = readmatrix('zipcode_train.csv');
    test = readmatrix('zipcode_test.csv');
    
    % training accuracy
    c_train = 0;
    for i = 1:length(train_class)
        if train_class(i) == train(i,end)
            c_train = c_train + 1;
        end
    end
    tra_acc = 100 * c_train / length(train_class);
    fprintf('The training accuracy for this model with lambda = %2.2f is %2.2f%%\n', lambda, tra_acc)
    
    % testing accuracy
    c_test = 0;
    for i = 1: length(test_class)
        if test_class(i) == test(i, end)
            c_test = c_test + 1;
        end
    end
    tst_acc = 100 * c_test / length(test_class);
    fprintf('The testing accuracy for this model with lambda = %2.2f is %2.2f%%\n\n', lambda, tst_acc)
end