function [Features] = getFeaturesVCLFER(path)
    load VCL_FER.mat

    numberOfImages = size(mos,1);

    net    = alexnet;
    Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

    numberOfImages = size(mos, 1);
    Scores = zeros(numberOfImages, 1);
    Features = zeros(numberOfImages, 1376);

    for i=1:numberOfImages
        %if(mod(i,50)==0)
        %    disp(i);
        %end
        tmp = Names{i};
        try
            imgDist = imread( strcat(path, filesep, tmp, '.bmp') );
        catch ME
            if( strcmp( ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist' ))
                imgDist = imread( strcat(path, filesep, tmp, '.jpg') );
            end
        end
        
        tmp2 = char(tmp);
        tmp2 = tmp2(1:6);
        
        imgRef = imread( strcat(path, filesep, tmp2, '.bmp') );
        
        Features(i,:) = getFeatures(imgDist, imgRef, Layers, net);
    end
end

