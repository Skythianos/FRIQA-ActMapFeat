clear all
close all

load TID2008_Data.mat

pathDistorted = 'C:\Users\Public\QualityAssessment\tid2008\distorted_images';
pathReference = 'C:\Users\Public\QualityAssessment\tid2008\reference_images';

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
            distortedImagePath = strcat(pathDistorted, filesep, distortedImageName);
            imgDist = imread(distortedImagePath);
        end
    end
    
    try
        imgRef  = imread(referenceImagePath);
    catch ME
        if( strcmp( ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist' ))
            referenceImagePath = strcat(pathReference, filesep, 'i25.bmp');
            imgRef  = imread(referenceImagePath);
        end
    end
    Features(i,:) = getFeatures(imgDist, imgRef, Layers, net);
end

PLCC = zeros(1,100); SROCC = zeros(1,100); KROCC = zeros(1,100);

parfor i=1:100
    rng(i);
    disp(i);
    [Train, Test] = splitTrainTest_TID2008(moswithnames);

    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);
    
    YTest = dmos(Test);
    YTrain= dmos(Train);

    Mdl = fitrsvm(TrainFeatures, YTrain, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    Pred= predict(Mdl,TestFeatures);
    
    eval = metric_evaluation(Pred, YTest);
    PLCC(i) = eval(1);
    SROCC(i)= eval(2);
    KROCC(i)= eval(3);
end

disp('----------------------------------');
X = ['Average PLCC after 100 random train-test splits: ', num2str(round(mean(PLCC(:)),3))];
disp(X);
X = ['Average SROCC after 100 random train-test splits: ', num2str(round(mean(SROCC(:)),3))];
disp(X);
X = ['Average KROCC after 100 random train-test splits: ', num2str(round(mean(KROCC(:)),3))];

figure;boxplot([PLCC',SROCC',KROCC'],{'PLCC','SROCC','KROCC'});
saveas(gcf,'TID2008_Box.png');
disp(X);