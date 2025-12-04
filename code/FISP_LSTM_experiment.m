

function results = FISP_LSTM_experiment(params)

    % Load prepared data (here: Germany). 
    %    For Italy, just load a different .mat file.
    S = load("TrainValidationTest.mat");
    %S = load("Germany_TrainValidationTest.mat", ...
     %        "XTrain","YTrain","XValidation","YValidation");
    XTrain      = S.XTrain;
    YTrain      = S.YTrain;
    XValidation = S.XValidation;
    YValidation = S.YValidation;

    % Network dimensions
    % Each sequence is [T x 1] → 1 feature, 1 output
    featureDimension = 1;
    numResponses     = 1;

    % Hyperparameters coming from Experiment Manager
    LSTMDepth      = round(params.LSTMDepth);  % integer, [1 3]
    LSTMDepth      = max(1, min(3, LSTMDepth)); % keep in [1,3] just in case

    numHiddenUnits = round(params.NumHiddenUnits);   % integer, [50 300]
    numHiddenUnits = max(50, min(300, numHiddenUnits));  % keep in [50,300]

    initLR         = params.InitialLearnRate; % real, [5e-4 1e-2]
    dropoutRate    = params.DropoutRate;      % real,  [0 0.6]

    % Define architecture
    layers = sequenceInputLayer(featureDimension);

    for i = 1:LSTMDepth
        layers = [layers; ...
            lstmLayer(numHiddenUnits, ...
                      OutputMode="sequence")];
    end

    layers = [layers
        fullyConnectedLayer(100)
        reluLayer
        dropoutLayer(dropoutRate)
        fullyConnectedLayer(numResponses)
        regressionLayer];

    % Training options
    maxEpochs     = 300;
    miniBatchSize = 20;

    options = trainingOptions("adam", ...
        ExecutionEnvironment   = "auto", ...
        MaxEpochs              = maxEpochs, ...
        MiniBatchSize          = miniBatchSize, ...
        ValidationData         = {XValidation, YValidation}, ...
        ValidationFrequency    = 20, ...
        InitialLearnRate       = initLR, ...
        LearnRateDropFactor    = 0.2, ...
        LearnRateDropPeriod    = 15, ...
        GradientThreshold      = 1, ...
        Shuffle                = "never", ...
        Verbose                = false, ...
        OutputFcn              = @FISP_epochProgress);

    disp("Starting trial with: Depth=" + LSTMDepth + ...
     ", Units=" + numHiddenUnits + ...
     ", LR=" + initLR + ...
     ", Dropout=" + dropoutRate);

    % Train
    [net, info] = trainNetwork(XTrain, YTrain, layers, options);
    

    % Validation MAE (our main "score" for this trial)
    YPred = predict(net, XValidation, MiniBatchSize=1);

    allAbsErr = [];
    for k = 1:numel(YValidation)
        absErr = abs(YPred{k} - YValidation{k});
        allAbsErr = [allAbsErr; absErr(:)];
    end
    valMAE = mean(allAbsErr);
    
    disp("Finished trial. Validation MAE = " + valMAE);

    % Results struct for Experiment Manager
    results.trainedNetwork = net;
    results.ValidationMAE  = valMAE;
    results.trainInfo      = info;
end
