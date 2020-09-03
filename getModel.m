function [Model] = getModel(Features, YTrain)
    Model = fitrsvm(Features, YTrain, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
end

