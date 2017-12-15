%%
preds = roData(:, 1:124);
preds = table2array(preds);
labels = strcmp(roData.output, 'rock');
fitglm(preds, labels, 'distribution', 'binomial')