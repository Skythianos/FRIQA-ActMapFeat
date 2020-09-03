function [Features] = getFeaturesCSIQ(pathDist, pathRef)
    load CSIQ.mat

    home = pwd;
    cd(pathDist);
    if ~exist('ALL', 'dir')
        mkdir 'ALL'
        copyfile awgn/* ALL
        copyfile blur/* ALL
        copyfile contrast/* ALL
        copyfile fnoise/* ALL
        copyfile jpeg/* ALL
        copyfile jpeg2000/* ALL
    else
        rmdir('ALL','s')
        mkdir 'ALL'
        copyfile awgn/* ALL
        copyfile blur/* ALL
        copyfile contrast/* ALL
        copyfile fnoise/* ALL
        copyfile jpeg/* ALL
        copyfile jpeg2000/* ALL
    end
    cd(home)
    pathDist = strcat(pathDist, filesep, 'ALL');

    dmos = cell2mat(dmos);
    numberOfImages = size(dmos,1);

    net    = alexnet;
    Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

    Scores = zeros(numberOfImages, 1);
    Features = zeros(numberOfImages, 1376);

    Filenames=cell(numberOfImages,1);

    parfor i=1:numberOfImages
        %disp(i);
        if(isnumeric(Image{i}))
            name = int2str(Image{i});
        else
            name = char(Image{i});
        end
    
        type = string(dst_type{i});
        if(strcmp(type,'noise'))
            dst = 'AWGN';
        elseif(strcmp(type,'jpeg'))
            dst = 'JPEG';
        elseif(strcmp(type,'jpeg 2000'))
            dst = 'jpeg2000';
        elseif(strcmp(type,'fnoise'))
            dst = 'fnoise';
        elseif(strcmp(type,'blur'))
            dst = 'BLUR';
        elseif(strcmp(type,'contrast'))
            dst = 'contrast';
        else
            error('Unknown distortion type'); 
        end
        level = int2str(dst_lev{i});
          
        filename = strcat(name, '.', dst, '.', level, '.png');
    
        Filenames{i}=filename;
    
        imgDist = imread(strcat(pathDist, filesep, filename));
        imgRef  = imread(strcat(pathRef,  filesep, num2str(Image{i}), '.png'));
    
        Features(i,:) = getFeatures(imgDist, imgRef, Layers, net); 
    end
end

