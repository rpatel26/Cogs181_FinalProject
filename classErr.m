function err = classErr(predArr, targetArr)
[rows, cols] = size(predArr);
wrong = 0;
for i = 1:rows
    if(predArr(i) ~= targetArr(i))
       wrong = wrong + 1;
    end
end

err = wrong/rows;

end