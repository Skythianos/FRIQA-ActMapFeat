# FRIQA-ActMapFeat
**Paper title**: A combined full-reference image quality assessment method based on convolutional activation maps<br/>
**Journal**:  Algorithms 2020, 13(12), 313; https://doi.org/10.3390/a13120313<br/> 
**Abstract**: The goal of full-reference image quality assessment (FR-IQA) is to predict the perceptual qualityof an image as perceived by human observers using its pristine (distortion free) reference counterpart. In this study, we explore a novel,  combined approach which predicts the perceptual quality of adistorted image by compiling a feature vector from convolutional activation maps. More specifically,a reference-distorted image pair is run through a pretrained convolutional neural network and theactivation maps are compared with a traditional image similarity metric. Subsequently, the resultingfeature vector is mapped onto perceptual quality scores with the help of a trained support vectorregressor. A detailed parameter study is also presented in which the design choices of the proposedmethod is explained. Furthermore, we study the relationship between the amount of training imagesand the prediction performance.  Specifically, it is demonstrated that the proposed method can betrained with a small amount of data to reach high prediction performance. Our best proposal—calledActMapFeat—is compared to the state-of-the-art on six publicly available benchmark IQA databases,such as KADID-10k, TID2013, TID2008, MDID, CSIQ, and VCL-FER. Specifically, our method is able tosignificantly outperform the state-of-the-art on these benchmark databases.<br/><br/>
This repository contains the source code belonging to "Varga, D. A Combined Full-Reference Image Quality Assessment Method Based on Convolutional Activation Maps. *Algorithms* **2020**, 13, 313."<br/>


The proposed method was tested on six publicly available image quality assessment databases. It produces the following results after 100 random train-test splits with respect to the reference images (appx. 80% of images for training, appx. 20% for testing).

|                |PLCC      |SROCC     |KROCC     |
|----------------|----------|----------|----------|
|KADID-10k       |0.959     |0.957     |0.819     |
|TID2013         |0.943     |0.936     |0.780     |
|TID2008         |0.941     |0.937     |0.790     |
|VCL-FER         |0.960     |0.961     |0.826     |
|MDID            |0.930     |0.927     |0.769     |
|CSIQ            |0.971     |0.970     |0.850     |

This repository contains the following MATLAB scripts:<br/>
1.) qualityCSIQ.m - demo script demonstrating the usage and results of the proposed method on CSIQ database <br/>
2.) qualityKADID.m - demo script demonstrating the usage and results of the proposed method on KADID-10k database <br/>
3.) qualityMDID.m - demo script demonstrating the usage and results of the proposed method on MDID database <br/>
4.) qualityTID2008.m - demo script demonstrating the usage and results of the proposed method on TID2008 database <br/>
5.) qualityTID2013.m - demo script demonstrating the usage and results of the proposed method on TID2013 database <br/>
6.) qualityVCLFER.m - demo script demonstrating the usage and results of the proposed method on VCL@FER database <br/>
7.) crossDatabaseTest.m - cross database test using CSIQ, KADID-10k, MDID, TID2013, TID2008, VCL@FER databases <br/>
8.) qualityKADID_DistLevels.m - evaluating KADID-10k database with respect to different distortion levels <br/>
9.) qualityKADID_DistTypes.m - evaluating KADID-10k database with respect to different distortion types <br/>
<br/>
<br/>
If you use this code, please cite the following paper: https://www.mdpi.com/1999-4893/13/12/313
