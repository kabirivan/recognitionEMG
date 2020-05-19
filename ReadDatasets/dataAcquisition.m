classdef dataAcquisition 
    %UNTITLED11 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        user
        version
        gesture
    end
    
    methods
        function obj = dataAcquisition(user,version,gesture)
            %UNTITLED11 Construct an instance of this class
            %   Detailed explanation goes here
            obj.user = user;
            obj.version = version;
            obj.gesture = gesture;
        end
        
        
        function [X, Y] = getTotalXnYByUser(obj)


        if (strcmp('training',obj.version) == 1)

            sampleType = 'trainingSamples';

        elseif (strcmp('testing',obj.version) == 1)

            sampleType = 'testingSamples';

        end


        numClasses = length(obj.gesture);
        X = cell(1, numClasses);
        Y = cell(1, numClasses);
        
        for class_i = 1:numClasses
            
            typeGesture = obj.gesture{class_i};
            gestureData = obj.user.(sampleType).(typeGesture);
            
            switch typeGesture
                case 'noGesture'
                    code = 1;
                case 'fist'
                    code = 2;
                case 'waveIn'
                    code = 3;
                case 'waveOut'
                    code = 4;
                case 'open'
                    code = 5;
                case 'pinch'
                    code = 6;
            end

            
            numTrialsForEachGesture = length(fieldnames(gestureData));
            x = cell(1, numTrialsForEachGesture);
            y = cell(1, numTrialsForEachGesture);
            

            for i_emg = 1:numTrialsForEachGesture

                sampleNum = sprintf('sample%d',i_emg);
                emgSample = gestureData.(sampleNum).emg;

                EMG = [];

                for ch = 1:8               
                    channel = sprintf('ch%d',ch); 
                    EMG(:,ch) = emgSample.(channel);
                end

                [samples, ~] = size(EMG);
                % GET X
                x{i_emg} = EMG;

                % GET Y
                y{i_emg} = repmat(code, samples, 1);
            end
            
           
            X{class_i} = x;
            Y{class_i} = y;
            
        end

        end
        
        
        
        
            
    end
end




