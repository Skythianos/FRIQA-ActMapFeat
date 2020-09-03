clear all
close all

disp('CSIQ Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathDist = '/home/domonkos/Desktop/QualityAssessment/Databases/CSIQ/dst_imgs';  % PATH CSIQ
pathRef  = '/home/domonkos/Desktop/QualityAssessment/Databases/CSIQ/src_imgs';  % PATH CSIQ
FeaturesCSIQ = getFeaturesCSIQ(pathDist, pathRef);

disp('TID2008 Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathDistorted = '/home/domonkos/Desktop/QualityAssessment/Databases/TID2008/tid2008/distorted_images'; % PATH TID2008
pathReference = '/home/domonkos/Desktop/QualityAssessment/Databases/TID2008/tid2008/reference_images'; % PATH TID2008
FeaturesTID2008 = getFeaturesTID2008(pathDistorted, pathReference);

disp('TID2013 Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathDistorted = '/home/domonkos/Desktop/QualityAssessment/Databases/TID2013/tid2013/distorted_images'; % PATH TID2013
pathReference = '/home/domonkos/Desktop/QualityAssessment/Databases/TID2013/tid2013/reference_images'; % PATH TID2013
[FeaturesTID2013] = getFeaturesTID2013(pathDistorted,pathReference);

disp('VCL@FER Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = '/home/domonkos/Desktop/QualityAssessment/Databases/VCL_FER/vcl_fer'; % PATH VCL-FER
[FeaturesVCLFER] = getFeaturesVCLFER(path);

disp('KADID-10k Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = '/home/domonkos/Desktop/QualityAssessment/Databases/kadid10k/images'; % PATH KADID
[FeaturesKADID10k] = getFeaturesKADID10k(path);

disp('MDID Feature Extraction');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathDist = '/home/domonkos/Desktop/QualityAssessment/Databases/MDID/distortion_images'; % PATH MDID
pathRef  = '/home/domonkos/Desktop/QualityAssessment/Databases/MDID/reference_images';  % PATH MDID
[FeaturesMDID] = getFeaturesMDID(pathDist, pathRef);

load CSIQ.mat
ModelTrainedCSIQ     = getModel(FeaturesCSIQ, cell2mat(dmos)); % Model trained on CSIQ

load TID2008_Data.mat
ModelTrainedTID2008  = getModel(FeaturesTID2008, dmos); % Model trained on TID2008

load TID2013_Data.mat
ModelTrainedTID2013  = getModel(FeaturesTID2013, dmos); % Model trained on TID2013

load VCL_FER.mat
ModelTrainedVCLFER   = getModel(FeaturesVCLFER, mos); % Model trained on VCL@FER

load KADID_Data2.mat
ModelTrainedKADID10k = getModel(FeaturesKADID10k, dmos); % Model trained on KADID-10k

load MDID.mat
ModelMDID = getModel(FeaturesMDID, mos); % Model trained on MDID

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Cross database testing model trained on KADID-10k');
load CSIQ.mat
P1 = predict(ModelTrainedKADID10k, FeaturesCSIQ);
PLCC=-corr(P1, cell2mat(dmos));
SROCC=-corr(P1, cell2mat(dmos),'Type','Spearman');
KROCC=-corr(P1, cell2mat(dmos),'Type','Kendall');
disp('Trained on KADID-10k, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P2 = predict(ModelTrainedKADID10k, FeaturesTID2008);
PLCC=corr(P2, dmos);
SROCC=corr(P2, dmos,'Type','Spearman');
KROCC=corr(P2, dmos,'Type','Kendall');
disp('Trained on KADID-10k, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P3 = predict(ModelTrainedKADID10k, FeaturesTID2013);
PLCC=corr(P3, dmos);
SROCC=corr(P3, dmos,'Type','Spearman');
KROCC=corr(P3, dmos,'Type','Kendall');
disp('Trained on KADID-10k, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P4 = predict(ModelTrainedKADID10k, FeaturesVCLFER);
PLCC=corr(P4, mos);
SROCC=corr(P4, mos,'Type','Spearman');
KROCC=corr(P4, mos,'Type','Kendall');
disp('Trained on KADID-10k, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load MDID.mat
P5 = predict(ModelTrainedKADID10k, FeaturesMDID);
PLCC=corr(P5, mos);
SROCC=corr(P5, mos,'Type','Spearman');
KROCC=corr(P5, mos,'Type','Kendall');
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
PLCC=-corr(P1, cell2mat(dmos));
SROCC=-corr(P1, cell2mat(dmos),'Type','Spearman');
KROCC=-corr(P1, cell2mat(dmos),'Type','Kendall');
disp('Trained on TID2013, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P2 = predict(ModelTrainedTID2013, FeaturesTID2008);
PLCC=corr(P2, dmos);
SROCC=corr(P2, dmos,'Type','Spearman');
KROCC=corr(P2, dmos,'Type','Kendall');
disp('Trained on TID2013, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelTrainedTID2013, FeaturesKADID10k);
PLCC=corr(P3, dmos);
SROCC=corr(P3, dmos,'Type','Spearman');
KROCC=corr(P3, dmos,'Type','Kendall');
disp('Trained on TID2013, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P4 = predict(ModelTrainedTID2013, FeaturesVCLFER);
PLCC=corr(P4, mos);
SROCC=corr(P4, mos,'Type','Spearman');
KROCC=corr(P4, mos,'Type','Kendall');
disp('Trained on TID2013, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load MDID.mat
P5 = predict(ModelTrainedTID2013, FeaturesMDID);
PLCC=corr(P5, mos);
SROCC=corr(P5, mos,'Type','Spearman');
KROCC=corr(P5, mos,'Type','Kendall');
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
PLCC=-corr(P1, cell2mat(dmos));
SROCC=-corr(P1, cell2mat(dmos),'Type','Spearman');
KROCC=-corr(P1, cell2mat(dmos),'Type','Kendall');
disp('Trained on VCL@FER, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P2 = predict(ModelTrainedVCLFER, FeaturesTID2008);
PLCC=corr(P2, dmos);
SROCC=corr(P2, dmos,'Type','Spearman');
KROCC=corr(P2, dmos,'Type','Kendall');
disp('Trained on VCL@FER, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelTrainedVCLFER, FeaturesKADID10k);
PLCC=corr(P3, dmos);
SROCC=corr(P3, dmos,'Type','Spearman');
KROCC=corr(P3, dmos,'Type','Kendall');
disp('Trained on VCL@FER, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P4 = predict(ModelTrainedVCLFER, FeaturesTID2013);
PLCC=corr(P4, dmos);
SROCC=corr(P4, dmos,'Type','Spearman');
KROCC=corr(P4, dmos,'Type','Kendall');
disp('Trained on VCL@FER, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load MDID.mat
P5 = predict(ModelTrainedVCLFER, FeaturesMDID);
PLCC=corr(P5, mos);
SROCC=corr(P5, mos,'Type','Spearman');
KROCC=corr(P5, mos,'Type','Kendall');
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
PLCC=-corr(P1, cell2mat(dmos));
SROCC=-corr(P1, cell2mat(dmos),'Type','Spearman');
KROCC=-corr(P1, cell2mat(dmos),'Type','Kendall');
disp('Trained on TID2008, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P2 = predict(ModelTrainedTID2008, FeaturesVCLFER);
PLCC=corr(P2, mos);
SROCC=corr(P2, mos,'Type','Spearman');
KROCC=corr(P2, mos,'Type','Kendall');
disp('Trained on TID2008, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelTrainedTID2008, FeaturesKADID10k);
PLCC=corr(P3, dmos);
SROCC=corr(P3, dmos,'Type','Spearman');
KROCC=corr(P3, dmos,'Type','Kendall');
disp('Trained on TID2008, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P4 = predict(ModelTrainedTID2008, FeaturesTID2013);
PLCC=corr(P4, dmos);
SROCC=corr(P4, dmos,'Type','Spearman');
KROCC=corr(P4, dmos,'Type','Kendall');
disp('Trained on TID2008, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load MDID.mat
P5 = predict(ModelTrainedTID2008, FeaturesMDID);
PLCC=corr(P5, mos);
SROCC=corr(P5, mos,'Type','Spearman');
KROCC=corr(P5, mos,'Type','Kendall');
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
PLCC=-corr(P1, cell2mat(dmos));
SROCC=-corr(P1, cell2mat(dmos),'Type','Spearman');
KROCC=-corr(P1, cell2mat(dmos),'Type','Kendall');
disp('Trained on MDID, Tested on CSIQ');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P2 = predict(ModelMDID, FeaturesVCLFER);
PLCC=corr(P2, mos);
SROCC=corr(P2, mos,'Type','Spearman');
KROCC=corr(P2, mos,'Type','Kendall');
disp('Trained on MDID, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelMDID, FeaturesKADID10k);
PLCC=corr(P3, dmos);
SROCC=corr(P3, dmos,'Type','Spearman');
KROCC=corr(P3, dmos,'Type','Kendall');
disp('Trained on MDID, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P4 = predict(ModelMDID, FeaturesTID2013);
PLCC=corr(P4, dmos);
SROCC=corr(P4, dmos,'Type','Spearman');
KROCC=corr(P4, dmos,'Type','Kendall');
disp('Trained on MDID, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P4 = predict(ModelMDID, FeaturesTID2008);
PLCC=corr(P4, dmos);
SROCC=corr(P4, dmos,'Type','Spearman');
KROCC=corr(P4, dmos,'Type','Kendall');
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
PLCC=-corr(P1, mos);
SROCC=-corr(P1, mos,'Type','Spearman');
KROCC=-corr(P1, mos,'Type','Kendall');
disp('Trained on CSIQ, Tested on MDID');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load VCL_FER.mat
P2 = predict(ModelTrainedCSIQ, FeaturesVCLFER);
PLCC=-corr(P2, mos);
SROCC=-corr(P2, mos,'Type','Spearman');
KROCC=-corr(P2, mos,'Type','Kendall');
disp('Trained on CSIQ, Tested on VCL@FER');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load KADID_Data2.mat
P3 = predict(ModelTrainedCSIQ, FeaturesKADID10k);
PLCC=-corr(P3, dmos);
SROCC=-corr(P3, dmos,'Type','Spearman');
KROCC=-corr(P3, dmos,'Type','Kendall');
disp('Trained on CSIQ, Tested on KADID-10k');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2013_Data.mat
P4 = predict(ModelTrainedCSIQ, FeaturesTID2013);
PLCC=-corr(P4, dmos);
SROCC=-corr(P4, dmos,'Type','Spearman');
KROCC=-corr(P4, dmos,'Type','Kendall');
disp('Trained on CSIQ, Tested on TID2013');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);

load TID2008_Data.mat
P4 = predict(ModelTrainedCSIQ, FeaturesTID2008);
PLCC=-corr(P4, dmos);
SROCC=-corr(P4, dmos,'Type','Spearman');
KROCC=-corr(P4, dmos,'Type','Kendall');
disp('Trained on CSIQ, Tested on TID2008');
X = ['PLCC: ', num2str(round(PLCC,3))];
disp(X);
X = ['SROCC: ', num2str(round(SROCC,3))];
disp(X);
X = ['KROCC: ', num2str(round(KROCC,3))];
disp(X);
