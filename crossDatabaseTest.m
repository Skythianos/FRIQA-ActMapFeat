clear all
close all

disp('CSIQ Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathDist = 'C:\Users\Public\QualityAssessment\CSIQ\dst_imgs';  % PATH CSIQ
pathRef  = 'C:\Users\Public\QualityAssessment\CSIQ\src_imgs';  % PATH CSIQ
FeaturesCSIQ = getFeaturesCSIQ(pathDist, pathRef);

disp('TID2008 Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathDistorted = 'C:\Users\Public\QualityAssessment\tid2008\distorted_images'; % PATH TID2008
pathReference = 'C:\Users\Public\QualityAssessment\tid2008\reference_images'; % PATH TID2008
FeaturesTID2008 = getFeaturesTID2008(pathDistorted, pathReference);

disp('TID2013 Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathDistorted = 'C:\Users\Public\QualityAssessment\tid2013\distorted_images'; % PATH TID2013
pathReference = 'C:\Users\Public\QualityAssessment\tid2013\reference_images'; % PATH TID2013
[FeaturesTID2013] = getFeaturesTID2013(pathDistorted,pathReference);

disp('VCL@FER Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = 'C:\Users\Public\QualityAssessment\VCL_FER\vcl_fer'; % PATH VCL-FER
[FeaturesVCLFER] = getFeaturesVCLFER(path);

disp('KADID-10k Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = 'C:\Users\Public\QualityAssessment\KADID-10k\images'; % PATH KADID
[FeaturesKADID10k] = getFeaturesKADID10k(path);

disp('MDID Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathDist = 'C:\Users\Public\QualityAssessment\MDID\distortion_images';
pathRef  = 'C:\Users\Public\QualityAssessment\MDID\reference_images';
[FeaturesMDID] = getFeaturesMDID(pathDist, pathRef);

load CSIQ.mat
disp('Train model on CSIQ Features');
ModelTrainedCSIQ     = getModel(FeaturesCSIQ, cell2mat(dmos));

load TID2008_Data.mat
disp('Train model on TID2008 Features');
ModelTrainedTID2008  = getModel(FeaturesTID2008, dmos);

load TID2013_Data.mat
disp('Train model on TID2013 Features');
ModelTrainedTID2013  = getModel(FeaturesTID2013, dmos);

load VCL_FER.mat
disp('Train model on VCL@FER Features');
ModelTrainedVCLFER   = getModel(FeaturesVCLFER, mos);

load KADID_Data2.mat
disp('Train model on KADID-10k Features');
ModelTrainedKADID10k = getModel(FeaturesKADID10k, dmos);

load MDID.mat
disp('Train model on MDID Features');
ModelMDID = getModel(FeaturesMDID, mos);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Cross database testing model trained on KADID-10k');
load CSIQ.mat
P1 = predict(ModelTrainedKADID10k, FeaturesCSIQ);
eval = metric_evaluation(P1, cell2mat(dmos));
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on KADID-10k, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P2 = predict(ModelTrainedKADID10k, FeaturesTID2008);
eval = metric_evaluation(P2, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on KADID-10k, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P3 = predict(ModelTrainedKADID10k, FeaturesTID2013);
eval = metric_evaluation(P3, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on KADID-10k, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P4 = predict(ModelTrainedKADID10k, FeaturesVCLFER);
eval = metric_evaluation(P4, mos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on KADID-10k, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load MDID.mat
P5 = predict(ModelTrainedKADID10k, FeaturesMDID);
eval = metric_evaluation(P5, mos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on KADID-10k, Tested on MDID');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Cross database testing model trained on TID2013');
load CSIQ.mat
P1 = predict(ModelTrainedTID2013, FeaturesCSIQ);
eval = metric_evaluation(P1, cell2mat(dmos));
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on TID2013, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P2 = predict(ModelTrainedTID2013, FeaturesTID2008);
eval = metric_evaluation(P2, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on TID2013, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelTrainedTID2013, FeaturesKADID10k);
eval = metric_evaluation(P3, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on TID2013, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P4 = predict(ModelTrainedTID2013, FeaturesVCLFER);
eval = metric_evaluation(P4, mos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on TID2013, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load MDID.mat
P5 = predict(ModelTrainedTID2013, FeaturesMDID);
eval = metric_evaluation(P5, mos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on TID2013, Tested on MDID');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Cross database testing model trained on VCL@FER');
load CSIQ.mat
P1 = predict(ModelTrainedVCLFER, FeaturesCSIQ);
eval = metric_evaluation(P1, cell2mat(dmos));
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on VCL@FER, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P2 = predict(ModelTrainedVCLFER, FeaturesTID2008);
eval = metric_evaluation(P2, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on VCL@FER, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelTrainedVCLFER, FeaturesKADID10k);
eval = metric_evaluation(P3, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on VCL@FER, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P4 = predict(ModelTrainedVCLFER, FeaturesTID2013);
eval = metric_evaluation(P4, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on VCL@FER, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load MDID.mat
P5 = predict(ModelTrainedVCLFER, FeaturesMDID);
eval = metric_evaluation(P5, mos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on VCL@FER, Tested on MDID');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Cross database testing model trained on TID2008');
load CSIQ.mat
P1 = predict(ModelTrainedTID2008, FeaturesCSIQ);
eval = metric_evaluation(P1, cell2mat(dmos));
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on TID2008, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P2 = predict(ModelTrainedTID2008, FeaturesVCLFER);
eval = metric_evaluation(P2, mos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on TID2008, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelTrainedTID2008, FeaturesKADID10k);
eval = metric_evaluation(P3, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on TID2008, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P4 = predict(ModelTrainedTID2008, FeaturesTID2013);
eval = metric_evaluation(P4, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on TID2008, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load MDID.mat
P5 = predict(ModelTrainedTID2008, FeaturesMDID);
eval = metric_evaluation(P5, mos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on TID2008, Tested on MDID');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Cross database testing model trained on MDID');
load CSIQ.mat
P1 = predict(ModelMDID, FeaturesCSIQ);
eval = metric_evaluation(P1, cell2mat(dmos));
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on MDID, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P2 = predict(ModelMDID, FeaturesVCLFER);
eval = metric_evaluation(P2, mos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on MDID, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelMDID, FeaturesKADID10k);
eval = metric_evaluation(P3, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on MDID, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P4 = predict(ModelMDID, FeaturesTID2013);
eval = metric_evaluation(P4, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on MDID, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P4 = predict(ModelMDID, FeaturesTID2008);
eval = metric_evaluation(P4, dmos);
PLCC=eval(1);
SROCC=eval(2);
KROCC=eval(3);
disp('Trained on MDID, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Cross database testing model trained on CSIQ');
load MDID.mat
P1 = predict(ModelTrainedCSIQ, FeaturesMDID);
eval = metric_evaluation(P1, mos);
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on CSIQ, Tested on MDID');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P2 = predict(ModelTrainedCSIQ, FeaturesVCLFER);
eval = metric_evaluation(P2, mos);
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on CSIQ, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelTrainedCSIQ, FeaturesKADID10k);
eval = metric_evaluation(P3, dmos);
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on CSIQ, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P4 = predict(ModelTrainedCSIQ, FeaturesTID2013);
eval = metric_evaluation(P4, dmos);
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on CSIQ, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P4 = predict(ModelTrainedCSIQ, FeaturesTID2008);
eval = metric_evaluation(P4, dmos);
PLCC=eval(1);
SROCC=-eval(2);
KROCC=-eval(3);
disp('Trained on CSIQ, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);