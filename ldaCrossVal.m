function [testErr, trainErr] = ldaCrossVal(data, N, K)
preds = data(:, 1:124);
labels = data.output;
preds = table2array(preds);
indices = crossvalind('Kfold', N, K);
testErr = 0;
trainErr = 0;
for i = 1:K
    test = (indices == i);
    train = ~test;
    trainPreds = preds(train);
    trainTarg = labels(train);
    testPreds = preds(test);
    testTarg = labels(test);
    
    testmdl = fitcdiscr(trainPreds, trainTarg, 'Prior', 'empirical');
    testYhat = predict(testmdl, testPreds);
    testErr = testErr + strClassErr(testYhat, testTarg);
    
    trainmdl = fitcdiscr(trainPreds, trainTarg, 'Prior', 'empirical');
    trainYhat = predict(trainmdl, trainPreds);
    trainErr = trainErr + strClassErr(trainYhat, trainTarg);
end
testErr = testErr/K;
trainErr = trainErr/K;
end