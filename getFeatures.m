function [output] = getFeatures(imgDist, imgRef, Layers, net)

    numberOfLayers = size(Layers, 2);
    output = [];
    
    for i=1:numberOfLayers
        activationsRef = activations(net, imgRef, Layers{i});
        activationsDist= activations(net, imgDist,Layers{i});
        
        depth = size(activationsRef,3);
        subVec= zeros(1, depth);
        
        for j=1:depth
            subVec(j) = HaarPSI(activationsRef(:,:,j), activationsDist(:,:,j)); 
        end
        
        output = [output, subVec];
    end
    
end

