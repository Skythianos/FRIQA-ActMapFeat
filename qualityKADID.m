clear all
close all

load KADID_Data2.mat % This mat file contains the names of images and MOS values

path = 'C:\Users\Public\QualityAssessment\KADID-10k\images'; % KADID-10k images (available: http://database.mmsp-kn.de/kadid-10k-database.html )

net    = alexnet;
Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

numberOfImages = size(dmos, 1);
Scores = zeros(numberOfImages, 1);
Features = zeros(numberOfImages, 1376);

parfor i=1:numberOfImages
    if(mod(i,1000)==0)
        disp(i);
    end
    imgDist  = imread( char(strcat(path, filesep, string(dist_img(i)))) );
    imgRef   = imread( char(strcat(path, filesep, string(ref_img(i)))) );
    Features(i,:) = getFeatures(imgDist, imgRef, Layers, net);
end

PLCC = zeros(1,20); SROCC = zeros(1,20); KROCC = zeros(1,20);

parfor i=1:20
    disp(i);
    [Train, Test] = splitTrainTest(dist_img);

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
