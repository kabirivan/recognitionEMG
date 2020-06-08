clc
clear all
close all
warning off all;

addpath('Preprocessing');
addpath('Segmentation');
addpath('DTW distance');
addpath('TrainingModel');
addpath('Feature extraction');

addpath('libs'); % libreria de Jonathan
gestures = {'noGesture', 'open', 'fist', 'waveIn', 'waveOut', 'pinch'};
predictions = [];
targets = [];
time = [];

%% ======================= Model Configuration ===========================

load options.mat
% This command makes possible the reproducibility of the results
rng('default'); 

%%
folderUserType = 'trainingJSON';
typeUser = dir(folderUserType);
numFiles = length(typeUser);
userProcessed = 0;


for user_i = 3
    
  if ~(strcmpi(typeUser(user_i).name, '.') || strcmpi(typeUser(user_i).name, '..') || strcmpi(typeUser(user_i).name, '.DS_Store'))

 %% Adquisition     
      
     userProcessed = userProcessed + 1;
     file = [folderUserType '/' typeUser(user_i).name];
     text = fileread(file);
     user = jsondecode(text);
     fprintf('Processing data from user: %d / %d\n', userProcessed, numFiles-2);
     close all;
    
    % Reading the training samples
     version = 'training'; 
     currentUserTrain = recognitionModel(user, version, gestures, options);
     [train_RawX_temp, train_Y_temp] = currentUserTrain.getTotalXnYByUser; 
    
%   %% Preprocessing   
%       % Filter applied  
%      train_FilteredX_temp = currentUserTrain.preProcessEMG(train_RawX_temp);
%       % Making a single set with the training samples of all the classes
%      [filteredDataX, dataY] = currentUserTrain.makeSingleSet(train_FilteredX_temp, train_Y_temp);
%      % Finding the EMG that is the center of each class
%      bestCenters = currentUserTrain.findCentersOfEachClass(filteredDataX, dataY);
%    %% Feature Extraction      
%      % Feature extraction by computing the DTW distanc
%      dataX = currentUserTrain.featureExtraction(filteredDataX, bestCenters);
%      % Preprocessing the feature vectors
%      nnModel = currentUserTrain.preProcessFeatureVectors(dataX);
%    
%    %% Training 
%      % Training the feed-forward NN
%      nnModel.model = currentUserTrain.trainSoftmaxNN(nnModel.dataX, dataY);
%      nnModel.numNeuronsLayers = currentUserTrain.numNeuronsLayers;
%      nnModel.transferFunctions = currentUserTrain.transferFunctions;
%      nnModel.centers = bestCenters;
%      
%     %% Testing  
%      % Reading the testing samples
%      version = 'testing';
%      currentUserTest = recognitionModel(user, version, gestures, options);  %%gestures 2 6
%      [test_RawX, test_Y] = currentUserTest.getTotalXnYByUser();
%      
%      % Classification
%      [predictedSeq, actualSeq, timeClassif, vectorTime] = currentUserTest.classifyEMGTraining_SegmentationNN(test_RawX, test_Y, nnModel);
%     
%      % Pos-processing labels
%      [predictedLabels, actualLabels, timePos] = currentUserTest.posProcessLabelsTraining(predictedSeq, actualSeq);
%      
%      % Concatenating the predictions of all the users for computing the
%      % errors
%      
%      recognitionResults.(user.userInfo.name).class = predictedLabels;
%      recognitionResults.(user.userInfo.name).vectorOfLabels = predictedSeq;
%      recognitionResults.(user.userInfo.name).vectorOfProcessingTimes = timeClassif;
%      recognitionResults.(user.userInfo.name).vectorOfTimePoints = vectorTime;
%      
%      responses.(user.userInfo.name) = currentUserTest.recognitionResults(predictedLabels,predictedSeq,timeClassif,vectorTime);   
     
  end
  
 % clc
end


% currentUserTest.plotConfusionMatrix(predictions,targets)
% 
% % Generate results for user
% %currentUserTest.generateResultsbyUser(responses);
% currentUserTest.generateResultsJSON(responses);

