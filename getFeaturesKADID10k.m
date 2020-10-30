function [Features] = getFeaturesKADID10k(path)
    dist_img = [];
    ref_img  = [];

    load KADID_Data2.mat % This mat file contains the names of images and MOS values

    net    = alexnet;
    Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

    numberOfImages = size(dmos, 1);
    Scores = zeros(numberOfImages, 1);
    Features = zeros(numberOfImages, 1376);

    parfor i=1:numberOfImages
        %if(mod(i,1000)==0)
        %    disp(i);
        %end
        imgDist  = imread( char(strcat(path, filesep, string(dist_img(i)))) );
        imgRef   = imread( char(strcat(path, filesep, string(ref_img(i)))) );
        Features(i,:) = getFeatures(imgDist, imgRef, Layers, net);
    end
end

