clear all
close all

load TID2013_Data.mat

pathDistorted = 'C:\Users\Public\QualityAssessment\tid2013\distorted_images';
pathReference = 'C:\Users\Public\QualityAssessment\tid2013\reference_images';

net    = alexnet;
Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

numberOfImages = size(dmos, 1);
Scores = zeros(numberOfImages, 1);
Features = zeros(numberOfImages, 1376);

parfor i=1:numberOfImages
    if(mod(i,100)==0)
        disp(i);
    end
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
            distortedImagePath = strcat(pathDistortedImage, filesep, distortedImageName);
            imgDist = imread(distortedImagePath);
        end
    end
    
    try
        imgRef  = imread(referenceImagePath);
    catch ME
        if( strcmp( ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist' ))
            imgRef = imread(strcat(pathReference, filesep, 'i25.bmp'));  
        end
    end
    Features(i,:) = getFeatures(imgDist, imgRef, Layers, net);
end

PLCC = zeros(1,20); SROCC = zeros(1,20); KROCC = zeros(1,20);

parfor i=1:20
    disp(i);
    [Train, Test] = splitTrainTest(moswithnames);

    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);
    
    YTest = dmos(Test);
    YTrain= dmos(Train);

    Mdl = fitrsvm(TrainFeatures, YTrain, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    Pred= predict(Mdl,TestFeatures);
    
    PLCC(i) = corr(Pred, YTest);
    SROCC(i)= corr(Pred, YTest, 'Type', 'Spearman');
    KROCC(i)= corr(Pred, YTest, 'Type', 'Kendall');
end

disp('----------------------------------');
X = ['Average PLCC after 20 random train-test splits: ', num2str(round(mean(PLCC(:)),3))];
disp(X);
X = ['Average SROCC after 20 random train-test splits: ', num2str(round(mean(SROCC(:)),3))];
disp(X);
X = ['Average KROCC after 20 random train-test splits: ', num2str(round(mean(KROCC(:)),3))];
disp(X);
