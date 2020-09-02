clear all
close all

load CSIQ.mat

home = pwd;
pathDist = '/home/dvi/Desktop/QualityAssessment/Databases/CSIQ/dst_imgs';
pathRef  = '/home/dvi/Desktop/QualityAssessment/Databases/CSIQ/src_imgs';

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
    disp(i);
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

Filenames = string(Filenames);

PLCC = zeros(1,20); SROCC = zeros(1,20); KROCC = zeros(1,20);

for i=1:20
    disp(i);
    [Train, Test] = splitTrainTest_CSIQ(Filenames);

    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);
    
    YTest = (dmos(Test))';
    YTrain= (dmos(Train))';

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