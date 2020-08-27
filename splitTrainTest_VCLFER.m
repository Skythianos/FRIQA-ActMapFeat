function [Train, Test] = splitTrainTest_VCLFER(names)

    numberOfImages = size(names,1);
   
    Train = false(numberOfImages,1);
    Test  = false(numberOfImages,1);
    
    p = randperm(23);
    
    train = p(1:round(23*0.80)); 
    
    for i=1:numberOfImages
        name = names{i};
        tmp = str2double(name(5:6));
        if( ismember(tmp,train) )
            Train(i)=true;
        else
            Test(i)=true;
        end
    end


end

