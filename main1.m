clc
clear all
close all

addpath('Preprocessing');
addpath('Segmentation');
addpath('DTW distance');
addpath('TrainingModel');
addpath('Feature extraction');
gestures = {'noGesture', 'open', 'fist', 'waveIn', 'waveOut', 'pinch'};
predictions = [];
recognition = [];
targets = [];
time = [];

%% ======================= Model Configuration ===========================

load options.mat



%%

fileTraining = dir('trainingJSON');
numFiles = length(fileTraining);
userProcessed = 0;


for user_i = 1:4
    
  if ~(strcmpi(fileTraining(user_i).name, '.') || strcmpi(fileTraining(user_i).name, '..') || strcmpi(fileTraining(user_i).name, '.DS_Store'))

    userProcessed = userProcessed + 1;
    file = ['trainingJSON/' fileTraining(user_i).name];
    text = fileread(file);
    user = jsondecode(text);
    fprintf('Processing data from user: %d / %d\n', userProcessed, numFiles-2);
    close all;
    
     version = 'training'; 
     currentUserTrain = recognitionModel(user, version, gestures, options);
    
     [train_RawX_temp, train_Y_temp] = currentUserTrain.getTotalXnYByUser; 
     train_FilteredX_temp = currentUserTrain.preProcessEMG(train_RawX_temp);
     [filteredDataX, dataY] = currentUserTrain.makeSingleSet(train_FilteredX_temp, train_Y_temp);
     bestCenters = currentUserTrain.findCentersOfEachClass(filteredDataX, dataY);
     
     dataX = currentUserTrain.featureExtraction(filteredDataX, bestCenters);
     nnModel = currentUserTrain.preProcessFeatureVectors(dataX);
   
    
    % Training the feed-forward NN
     nnModel.model = currentUserTrain.trainSoftmaxNN(nnModel.dataX, dataY);
     nnModel.numNeuronsLayers = currentUserTrain.numNeuronsLayers;
     nnModel.transferFunctions = currentUserTrain.transferFunctions;
     nnModel.centers = bestCenters;
   
     version = 'testing';
     currentUserTest = recognitionModel(user, version, gestures(2:end), options);
     [test_RawX, test_Y] = currentUserTest.getTotalXnYByUser; 
     [predictedSeq, actualSeq, timeClassif, vectorTime] = currentUserTest.classifyEMG_SegmentationNN(test_RawX, test_Y, nnModel);
     [predictedLabels, actualLabels, timePos] = currentUserTest.posProcessLabels(predictedSeq, actualSeq);
 
     predictions = [predictions, predictedLabels];
     targets = [targets, actualLabels];

     recognitionResults.(user.userInfo.name) = currentUserTest.recognitionResults(predictedLabels,predictedSeq,timeClassif,vectorTime);  

     
  end
  
  
end


