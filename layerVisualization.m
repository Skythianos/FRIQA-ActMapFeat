clear all
close all

imgRef = imread('peppers.png');
imgDist= imnoise(imgRef, 'gaussian', 0.25);

net = alexnet;

actRef = activations(net, imgRef, 'conv1');

sz = size(actRef);
actRef = reshape(actRef,[sz(1) sz(2) 1 sz(3)]);

IRef = imtile(mat2gray(actRef),'GridSize',[10 10]);

actDist= activations(net, imgDist, 'conv1');

sz = size(actDist);
actDist= reshape(actDist,[sz(1) sz(2) 1 sz(3)]);

IDist= imtile(mat2gray(actDist),'GridSize',[10 10]);

figure;
imshow(IRef);
title('Reference');

figure;
imshow(IDist);
title('Distorted');