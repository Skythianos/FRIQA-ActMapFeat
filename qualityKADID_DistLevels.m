clear all
close all

load KADID_Data2.mat % This mat file contains the names of images and MOS values

path = 'C:\Users\Public\QualityAssessment\KADID-10k\images';

numberOfImages = size(dmos, 1);
Scores = zeros(numberOfImages, 1);
Features = zeros(numberOfImages, 1376);

net    = alexnet;
Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

parfor i=1:numberOfImages
    if(mod(i,500)==0)
        disp(i);
    end
    imgDist  = imread( char(strcat(path, filesep, string(dist_img(i)))) );
    imgRef   = imread( char(strcat(path, filesep, string(ref_img(i)))) );
    Features(i,:) = getFeatures(imgDist, imgRef, Layers, net);
end

numSplit = 100;

Level=zeros(5,numSplit,3);

for i=1:numSplit
    disp(i);
    rng(i);
    [Train, Test, ~, Levels] = splitTrainTest_2(dist_img);
    
    test_img = dist_img(Test);

    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);
    
    YTest = dmos(Test);
    YTrain= dmos(Train);

    Mdl =fitrsvm(TrainFeatures,YTrain,'KernelFunction','gaussian','KernelScale','auto','Standardize', true);
    Pred=predict(Mdl,TestFeatures);
    
    for j=1:5
        test = dmos(Levels(:,j));
        selected = searchLevel(test_img, j);
        pred = Pred(selected);
        Level(j,i,1) = corr(pred , test);
        Level(j,i,2) = corr(pred , test, 'Type', 'Spearman');
        Level(j,i,3) = corr(pred , test, 'Type', 'Kendall');
    end
end

X = ['Level 1 - PLCC: ', num2str(round(mean(Level(1,:,1)),3))]; disp(X);
X = ['Level 1 - SROCC: ', num2str(round(mean(Level(1,:,2)),3))]; disp(X);
X = ['Level 1 - KROCC: ', num2str(round(mean(Level(1,:,3)),3))]; disp(X);
disp('-----------------------------------------------');
X = ['Level 2 - PLCC: ', num2str(round(mean(Level(2,:,1)),3))]; disp(X);
X = ['Level 2 - SROCC: ', num2str(round(mean(Level(2,:,2)),3))]; disp(X);
X = ['Level 2 - KROCC: ', num2str(round(mean(Level(2,:,3)),3))]; disp(X);
disp('-----------------------------------------------');
X = ['Level 3 - PLCC: ', num2str(round(mean(Level(3,:,1)),3))]; disp(X);
X = ['Level 3 - SROCC: ', num2str(round(mean(Level(3,:,2)),3))]; disp(X);
X = ['Level 3 - KROCC: ', num2str(round(mean(Level(3,:,3)),3))]; disp(X);
disp('-----------------------------------------------');
X = ['Level 4 - PLCC: ', num2str(round(mean(Level(4,:,1)),3))]; disp(X);
X = ['Level 4 - SROCC: ', num2str(round(mean(Level(4,:,2)),3))]; disp(X);
X = ['Level 4 - KROCC: ', num2str(round(mean(Level(4,:,3)),3))]; disp(X);
disp('-----------------------------------------------');
X = ['Level 5 - PLCC: ', num2str(round(mean(Level(5,:,1)),3))]; disp(X);
X = ['Level 5 - SROCC: ', num2str(round(mean(Level(5,:,2)),3))]; disp(X);
X = ['Level 5 - KROCC: ', num2str(round(mean(Level(5,:,3)),3))]; disp(X);
disp('-----------------------------------------------');