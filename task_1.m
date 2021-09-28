% Main for task 1
% Apply the regression to the training and testing data
y_train = proj_2_regress('regression.tra.csv', 'regression.tra.csv', 8);
y_test = proj_2_regress('regression.tra.csv', 'regression.tst.csv', 8);

% Training and testing error
train = readmatrix('regression.tra.csv');
test = readmatrix('regression.tst.csv');
train_in = train(:,1:8);
train_out = train(:,9:15);
test_in = test(:,1:8);
test_out = test(:,9:15);

E_train_mean  = 100 * mean(abs(y_train - train_out) ./ train_out);
E_test_mean = 100 * mean(abs(y_test - test_out) ./ test_out);
E_train_max = 100 * max(abs(y_train - train_out) ./ train_out);
E_test_max = 100 * max(abs(y_test - test_out) ./ test_out);
E_train_std = 100 * std(abs(y_train - train_out) ./ train_out);
E_test_std = 100 * std(abs(y_test - test_out) ./ test_out);

% Print restults
disp('The mean training error of the model is:')
disp('Output_1   Output_2    Output_3    Output_4    Output_5   Output_6    Output_7')
fprintf('%2.2f%%      ', E_train_mean)
fprintf('\n\n')
disp('The standard deviation of the training error of the model is:')
disp('Output_1   Output_2    Output_3    Output_4    Output_5    Output_6    Output_7')
fprintf('%2.2f%%      ', E_train_std)
fprintf('\n\n')
disp('The max training error of the model is:')
disp('Output_1    Output_2     Output_3     Output_4     Output_5     Output_6     Output_7')
fprintf('%2.2f%%      ', E_train_max)
fprintf('\n\n')
disp('The mean testing error of the model is:')
disp('Output_1   Output_2    Output_3    Output_4    Output_5    Output_6    Output_7')
fprintf('%2.2f%%      ', E_test_mean)
fprintf('\n\n')
disp('The standard deviation of the testing error of the model is:')
disp('Output_1   Output_2    Output_3    Output_4    Output_5    Output_6    Output_7')
fprintf('%2.2f%%      ', E_test_std)
fprintf('\n\n')
disp('The max testing error of the model is:')
disp('Output_1   Output_2     Output_3     Output_4     Output_5     Output_6     Output_7')
fprintf('%2.2f%%      ', E_test_max)
fprintf('\n')