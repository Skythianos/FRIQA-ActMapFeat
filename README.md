# FRIQA-ActMapFeat
Full-reference image quality assessment based on convolutional activation maps.

The proposed method was tested on six publicly available image quality assessment databases. It produces the following results after 100 random train-test splits with respect to the reference images (appx. 80% of images for training, appx. 20% for testing).

|                |PLCC      |SROCC     |KROCC     |
|----------------|----------|----------|----------|
|KADID-10k       |0.959     |0.957     |0.819     |
|TID2013         |0.943     |0.936     |0.780     |
|TID2008         |0.945     |0.943     |0.798     |
|VCL-FER         |0.957     |0.961     |0.826     |
|MDID            |0.923     |0.923     |0.765     |
|CSIQ            |0.968     |0.969     |0.848     |
