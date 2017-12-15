function [testErr, trainErr] = crossVal(data, N, K, p, gen)
shuffData = data(randperm(size(data, 1)), :);
preds = shuffData(:, 1:124);
preds = table2array(preds);
labels = strcmp(shuffData.output, gen);
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
    
    testmdl = fitglm(trainPreds, trainTarg, 'distribution', 'binomial');
    testYhat = predict(testmdl, testPreds);
    classedTestYhat = classifier(testYhat, p);
    testErr = testErr + classErr(classedTestYhat, testTarg);
    
    trainmdl = fitglm(trainPreds, trainTarg, 'distribution', 'binomial');
    trainYhat = predict(trainmdl, trainPreds);
    classedTrainYhat = classifier(trainYhat, p);
    trainErr = trainErr + classErr(classedTrainYhat, trainTarg);
end
testErr = testErr/K;
trainErr = trainErr/K;
end