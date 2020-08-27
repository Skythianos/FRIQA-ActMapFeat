clear all
close all

load MDID.mat

pathDist = 'C:\Users\Public\QualityAssessment\MDID\distortion_images';
pathRef  = 'C:\Users\Public\QualityAssessment\MDID\reference_images';

S = dir(fullfile(pathDist, '*.bmp'));

net    = alexnet;
Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

numberOfImages = size(mos, 1);
Scores = zeros(numberOfImages, 1);
Features = zeros(numberOfImages, 1376);

parfor i=1:numberOfImages
    if(mod(i,100)==0)
        disp(i);
    end
    F = fullfile(pathDist, S(i).name);
    refImgName = strcat(S(i).name(1:5), '.bmp');
    
    DistortedImg = imread(F);
    ReferenceImg = imread(strcat(pathRef, filesep, refImgName));
    
    Features(i,:) = getFeatures(DistortedImg, ReferenceImg, Layers, net);
end

names = string(cell2mat(struct2cell(struct('name', {S(1:end).name}))));

PLCC = zeros(1,20); SROCC = zeros(1,20); KROCC = zeros(1,20);

for i=1:20
    disp(i);
    [Train, Test] = splitTrainTest_MDID(names);

    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);
    
    YTest = (mos(Test))';
    YTrain= (mos(Train))';

    Mdl = fitrsvm(TrainFeatures, YTrain, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    Pred= predict(Mdl,TestFeatures);
    
    PLCC(i) = corr(Pred, YTest');
    SROCC(i)= corr(Pred, YTest', 'Type', 'Spearman');
    KROCC(i)= corr(Pred, YTest', 'Type', 'Kendall');
end

disp('----------------------------------');
X = ['Average PLCC after 20 random train-test splits: ', num2str(round(mean(PLCC(:)),3))];
disp(X);
X = ['Average SROCC after 20 random train-test splits: ', num2str(round(mean(SROCC(:)),3))];
disp(X);
X = ['Average KROCC after 20 random train-test splits: ', num2str(round(mean(KROCC(:)),3))];
disp(X);
