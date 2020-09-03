function [Features] = getFeaturesTID2013(pathDistorted,pathReference)
    load TID2013_Data.mat

    net    = alexnet;
    Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

    numberOfImages = size(dmos, 1);
    Scores = zeros(numberOfImages, 1);
    Features = zeros(numberOfImages, 1376);

    parfor i=1:numberOfImages
        %if(mod(i,100)==0)
        %    disp(i);
        %end
        distortedImageName = moswithnames{i};
        distortedImagePath = strcat(pathDistorted, filesep, distortedImageName);
    
        tmp = char(distortedImageName);
        tmp = upper(tmp(1:3));
        tmp = string(tmp);
    
        referenceImageName = strcat(tmp,'.BMP');
        referenceImagePath = strcat(pathReference, filesep, referenceImageName);
    
        try
            imgDist = imread(distortedImagePath);
        catch ME
            if( strcmp( ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist' ))
                distortedImageName(1) = 'I';
                distortedImagePath = strcat(pathDistorted, filesep, distortedImageName);
                imgDist = imread(distortedImagePath);
            end
        end
    
        try
            imgRef  = imread(referenceImagePath);
        catch ME
            if( strcmp( ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist' ))
                disp(referenceImagePath);
                referenceImagePath = strcat(pathReference, filesep, 'i25.bmp');
                imgRef  = imread(referenceImagePath);
            end
        end
        Features(i,:) = getFeatures(imgDist, imgRef, Layers, net);
    end
end

