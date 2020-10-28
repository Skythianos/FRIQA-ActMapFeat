clear all
close all

load VCL_FER.mat

path = '/home/domonkos/Desktop/QualityAssessment/Databases/VCL_FER/vcl_fer';

numberOfImages = size(mos,1);

net    = alexnet;
Layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

numberOfImages = size(mos, 1);
Scores = zeros(numberOfImages, 1);
Features = zeros(numberOfImages, 1376);

for i=1:numberOfImages
    if(mod(i,50)==0)
        disp(i);
    end
    tmp = Names{i};
    try
        imgDist = imread( strcat(path, filesep, tmp, '.bmp') );
    catch ME
        if( strcmp( ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist' ))
            imgDist = imread( strcat(path, filesep, tmp, '.jpg') );
        end
    end
        
    tmp2 = char(tmp);
    tmp2 = tmp2(1:6);
        
    imgRef = imread( strcat(path, filesep, tmp2, '.bmp') );
        
    Features(i,:) = getFeatures(imgDist, imgRef, Layers, net);
end

Names=string(Names);

PLCC = zeros(1,100); SROCC = zeros(1,100); KROCC = zeros(1,100);

for i=1:100
    disp(i);
    rng(i);
    [Train, Test] = splitTrainTest_VCLFER(Names);

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
X = ['Average PLCC after 100 random train-test splits: ', num2str(round(mean(PLCC(:)),3))];
disp(X);
X = ['Average SROCC after 100 random train-test splits: ', num2str(round(mean(SROCC(:)),3))];
disp(X);
X = ['Average KROCC after 100 random train-test splits: ', num2str(round(mean(KROCC(:)),3))];
disp(X);

figure;boxplot([PLCC',SROCC',KROCC'],{'PLCC','SROCC','KROCC'});
saveas(gcf,'VCLFER_Box.png');
