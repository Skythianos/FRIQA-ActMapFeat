function [Train,Test,Distortion,Level] = splitTrainTest_2(Names)

    numberOfImages = size(Names,1);
   
    Train = false(numberOfImages,1);
    Test  = false(numberOfImages,1);
    
    Distortion = false(numberOfImages,25);
    Level      = false(numberOfImages, 5);
    
    p = randperm(81);
    
    train = p(1:round(81*0.80)); 
    
    for i=1:numberOfImages
        name = Names{i};
        tmp1 = char(name);
        tmp2 = str2double(tmp1(2:3));
        
        dist = str2double(tmp1(5:6));
        level = str2double(tmp1(8:9));
        
        if( ismember(tmp2,train) )
            Train(i)=true;
        else
            Test(i)=true;
            Distortion(i,dist)=true;
            Level(i,level)=true;
        end
    end

end

