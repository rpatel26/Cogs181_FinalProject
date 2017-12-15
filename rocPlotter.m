function goodClass = rocPlotter(data, lowThresh, highThresh)
newdata = data(:, 3:11);
datarray = table2array(newdata);
predictors = datarray(:, 1:8);
targets = datarray(:, 9);
mdl = fitglm(predictors, targets, 'distribution', 'binomial');
dataPreds = predict(mdl, predictors);
goodClass = [];
hold on
for i = 0:100
    class = classifier(dataPreds, i/100);
    conMat = confusionmat(targets, class);
    scatter(conMat(1,2) / (conMat(1,2) + conMat(1,1)), conMat(2,2) / (conMat(2,2) + conMat(2,1)), 'bo')
    if(conMat(1,2) / (conMat(1,2) + conMat(1,1)) < lowThresh && conMat(2,2) / (conMat(2,2) + conMat(2,1)) > highThresh)
        goodClass = [goodClass, i/100];
    end
    if(i == 57)
       txt1 = '\leftarrow Threshold of 0.57';
       text(conMat(1,2) / (conMat(1,2) + conMat(1,1)), conMat(2,2) / (conMat(2,2) + conMat(2,1)), txt1) 
    end
end
title 'Stock Buying Thresholds';
xlabel('False Positive Rate')
ylabel('True Positive Rate')
aline = refline(0, 0.9);
aline.Color = 'r';
cline = refline(0, 0.44);
cline.Color = 'b';
legend([aline; cline], 'Aggressive', 'Cautious')
hold off
end