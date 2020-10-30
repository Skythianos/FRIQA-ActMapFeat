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

Dist=zeros(25,numSplit,3); 

for i=1:numSplit
    disp(i);
    rng(i);
    [Train, Test, Distortions, ~] = splitTrainTest_2(dist_img);
    
    test_img = dist_img(Test);

    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);
    
    YTest = dmos(Test);
    YTrain= dmos(Train);

    Mdl = fitrsvm(TrainFeatures,YTrain,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);
    %Mdl = fitrgp(TrainFeatures,YTrain,'KernelFunction','rationalquadratic','Standardize',true);
    Pred= predict(Mdl,TestFeatures);
    
    for j=1:25
        test = dmos(Distortions(:,j));
        selected = searchDistortion(test_img, j);
        pred = Pred(selected);
        Dist(j,i,1) = corr(pred , test);
        Dist(j,i,2) = corr(pred , test, 'Type', 'Spearman');
        Dist(j,i,3) = corr(pred , test, 'Type', 'Kendall');
    end
end

for j=1:25
    X = ['Type ', num2str(j), ' - PLCC: ', num2str(round(mean(Dist(j,:,1)),3))]; disp(X);
    X = ['Type ', num2str(j), ' - SROCC: ', num2str(round(mean(Dist(j,:,2)),3))]; disp(X);
    X = ['Type ', num2str(j), ' - KROCC: ', num2str(round(mean(Dist(j,:,3)),3))]; disp(X);
    disp('-----------------------------------------------');
end

disp('Information about distortion types can be found: http://database.mmsp-kn.de/kadid-10k-database.html');