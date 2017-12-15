function err = strClassErr(predArr, targetArr)
[rows, cols] = size(predArr);
wrong = 0;
for i = 1:rows
    if(strcmp(predArr(i), targetArr(i)) == 0)
       wrong = wrong + 1;
    end
end

err = wrong/rows;

end