function [selected] = searchDistortion(test_img, j)

    numberOfImages = size(test_img,1);
    
    selected = false(numberOfImages,1);

    for i=1:numberOfImages
        name = test_img{i};
        tmp1 = char(name);
                
        dist = str2double(tmp1(5:6));
        
        if( dist==j )
            selected(i)=true;
        else
            
        end
    end
    
end

