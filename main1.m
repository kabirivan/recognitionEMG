clc
clear all
close all

addpath('Preprocessing');
addpath('Segmentation');
addpath('DTW distance');
addpath('TrainingModel');
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
    
    % Reading the training samples
     version = 'training'; 
     currentUserTrain = recognitionModel(user, version, gestures, options);
    
     % Preprocessing
     [train_RawX_temp, train_Y_temp] = currentUserTrain.getTotalXnYByUser; 
     train_FilteredX_temp = currentUserTrain.preProcessEMG(train_RawX_temp);
     
      % Making a single set with the training samples of all the classes
     [filteredDataX, dataY] = currentUserTrain.makeSingleSet(train_FilteredX_temp, train_Y_temp);
     
     % Finding the EMG that is the center of each class
     bestCenters = currentUserTrain.findCentersOfEachClass(filteredDataX, dataY);
     
     % Feature extraction by computing the DTW distanc
     dataX = currentUserTrain.featureExtraction(filteredDataX, bestCenters);
     
     % Preprocessing the feature vectors
     nnModel = currentUserTrain.preProcessFeatureVectors(dataX);
   
    
     % Training the feed-forward NN
     nnModel.model = currentUserTrain.trainSoftmaxNN(nnModel.dataX, dataY);
     nnModel.numNeuronsLayers = currentUserTrain.numNeuronsLayers;
     nnModel.transferFunctions = currentUserTrain.transferFunctions;
     nnModel.centers = bestCenters;
     
     % Reading the testing samples
     version = 'testing';
     currentUserTest = recognitionModel(user, version, gestures, options);
     [test_RawX, test_Y] = currentUserTest.getTotalXnYByUser; 
     
     % Classification
     [predictedSeq, actualSeq, timeClassif, vectorTime] = currentUserTest.classifyEMG_SegmentationNN(test_RawX, test_Y, nnModel);
     
     % Pos-processing labels
     [predictedLabels, actualLabels, timePos] = currentUserTest.posProcessLabels(predictedSeq, actualSeq);
     
     % Concatenating the predictions of all the users for computing the
     % errors
     predictions = [predictions, predictedLabels];
     targets = [targets, actualLabels];
     recognitionResults.(user.userInfo.name) = currentUserTest.recognitionResults(predictedLabels,predictedSeq,timeClassif,vectorTime);  

     % Computing the time of processing
     estimatedTime = currentUserTest.computeTime(timeClassif, timePos);
     time = [time, estimatedTime];
     
     
  end
  
  
end

% Computing the confusion matrix and time of processing
currentUserTest.plotConfusionMatrix(predictions,targets,time);


