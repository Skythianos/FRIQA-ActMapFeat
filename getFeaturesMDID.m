function [Features] = getFeaturesMDID(pathDist, pathRef)
    S = [];
    load MDID.mat
    
    S = dir(fullfile(pathDist, '*.bmp'));

    net    = alexnet;
    Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

    numberOfImages = size(mos, 1);
    Scores = zeros(numberOfImages, 1);
    Features = zeros(numberOfImages, 1376);

    parfor i=1:numberOfImages
        %if(mod(i,100)==0)
        %    disp(i);
        %end
        F = fullfile(pathDist, S(i).name);
        refImgName = strcat(S(i).name(1:5), '.bmp');
    
        DistortedImg = imread(F);
        ReferenceImg = imread(strcat(pathRef, filesep, refImgName));
    
        Features(i,:) = getFeatures(DistortedImg, ReferenceImg, Layers, net);
    end

end

