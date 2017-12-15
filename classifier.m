function retArr = classifier(arr, p)
[inRows, inCols] = size(arr);
retArr = zeros(inRows, inCols);

for i = 1:inRows
    for j = 1:inCols
        if(arr(i, j) > p)
            retArr(i, j) = 1;
        else
            retArr(i, j) = 0;
        end
    end
end
end