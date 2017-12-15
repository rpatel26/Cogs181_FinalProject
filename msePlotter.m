function minErr = msePlotter(data, gen)
hold on
minErr = [];
for i = 0:100
    [testErr, trainErr] = crossval(data, 1372, 10, i/100, gen);
    minErr = [minErr, testErr];
    scatter(i/100, testErr)
end
xlabel('Classification Probability')
ylabel('Testing Error')
hold off
end