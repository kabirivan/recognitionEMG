clc
clear all
close all
warning off all;

% Xavier Aguas.
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% xavier.aguas@epn.edu.ec
% Jun 5, 2020


addpath('ReadDataset');
addpath('Preprocessing');
addpath('Segmentation');
addpath('DTW distance');
addpath('TrainingModel');
addpath('Feature extraction');

addpath('libs'); % libreria de Jonathan
gestures = {'noGesture', 'open', 'fist', 'waveIn', 'waveOut', 'pinch'};



%% ======================= Model Configuration ===========================

load options1.mat


% This command makes possible the reproducibility of the results
rng('default'); 

%%
userFolder = 'training';
folderData = [userFolder 'JSON'];
filesInFolder = dir(folderData);
numFiles = length(filesInFolder);
userProcessed = 0;
responses.userGroup = userFolder; 
gestures = {'noGesture', 'open', 'fist', 'waveIn', 'waveOut', 'pinch'};

for user_i = 1:numFiles
    
  if ~(strcmpi(filesInFolder(user_i).name, '.') || strcmpi(filesInFolder(user_i).name, '..') || strcmpi(filesInFolder(user_i).name, '.DS_Store'))

 %% Adquisition     
      
     userProcessed = userProcessed + 1;
     file = [folderData '/' filesInFolder(user_i).name];
     text = fileread(file);
     user = jsondecode(text);
     fprintf('Processing data from user: %d / %d\n', userProcessed, numFiles-3);
     close all;
    
    % Reading the training samples
     version = 'training'; 
     currentUserTrain = recognitionModel(user, version, gestures, options);
     [train_RawX_temp, train_Y_temp] = currentUserTrain.getTotalXnYByUser(); 
    
   %% Preprocessing   
       % Filter applied  
     train_FilteredX_temp = currentUserTrain.preProcessEMG(train_RawX_temp);
       % Making a single set with the training samples of all the classes
     [filteredDataX, dataY] = currentUserTrain.makeSingleSet(train_FilteredX_temp, train_Y_temp);
      % Finding the EMG that is the center of each class
      bestCenters = currentUserTrain.findCentersOfEachClass(filteredDataX, dataY);
    %% Feature Extraction      
      % Feature extraction by computing the DTW distanc
      dataX = currentUserTrain.featureExtraction(filteredDataX, bestCenters);
      % Preprocessing the feature vectors
      nnModel = currentUserTrain.preProcessFeatureVectors(dataX);
    
    %% Training 
      % Training the feed-forward NN
      nnModel.model = currentUserTrain.trainSoftmaxNN(nnModel.dataX, dataY);
      nnModel.numNeuronsLayers = currentUserTrain.numNeuronsLayers;
      nnModel.transferFunctions = currentUserTrain.transferFunctions;
      nnModel.centers = bestCenters;
      
     %% Testing  
      % Reading the testing samples
      version = 'testing';
      responses.repGroup = version;
      currentUserTest = recognitionModel(user, version, gestures, options);  %%gestures 2 6
      test_RawX = currentUserTest.getTotalXnYByUser();
      
      % Classification
      [predictedSeq,  timeClassif, vectorTime] = currentUserTest.classifyEMG_SegmentationNN(test_RawX, nnModel);
     
      % Pos-processing labels
      predictedLabels = currentUserTest.posProcessLabels(predictedSeq);
      
      % Concatenating the predictions of all the users for computing the
      % errors
      
      responses.userSet.(user.userInfo.name) = currentUserTest.recognitionResults(predictedLabels,predictedSeq,timeClassif,vectorTime,'testing');   
      
      %responses.(user.userInfo.name) = currentUserTest.recognitionResults2(predictedLabels,predictedSeq,timeClassif,vectorTime,'testing'); 
  end
  
  clc
end

% currentUserTest.generateResultsJSON(responses);

