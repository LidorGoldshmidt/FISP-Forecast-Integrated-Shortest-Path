%% Functions
%% Creating dataset

function [total_data_matrix, days_per_year] = buildWeekdaySeasonMatrixFromTime( ...
        data_files, timeColName, loadColName, samplesPerDay, initial_month, final_month)

    total_data_matrix = [];
    days_per_year     = zeros(numel(data_files), 1);

    for year_ind = 1:numel(data_files)
        file = string(data_files(year_ind));
        T    = readtable(file);

        % Extract time and load columns 
        t_raw = T.(timeColName);   
        
        if iscell(t_raw)
            t_raw = string(t_raw);
        end
        
        t_start = extractBefore(t_raw, " -");  
        
        % Parse using day/month/year and HH:mm format
        t = datetime(t_start, ...
                     'InputFormat','dd/MM/yyyy HH:mm', ...
                     'TimeZone','UTC');   


        y = T.(loadColName);  % Actual total load [MW]

        % Filter by desired months and weekdays (Mon–Fri) 
        m  = month(t);
        wd = weekday(t);   % 1=Sunday, 2=Monday, ..., 7=Saturday

        isDesiredMonth   = (m >= initial_month) & (m <= final_month);
        isWeekday        = (wd >= 2) & (wd <= 6);  % Monday–Friday

        idxKeep = isDesiredMonth & isWeekday;

        t_sel = t(idxKeep);
        y_sel = y(idxKeep);

        %  Group by calendar day 
        dayStamp = dateshift(t_sel, 'start', 'day'); % dateshift gives the start of each day (00:00)
        [uniqueDays, ~, dayIdx] = unique(dayStamp);

        data_matrix_year = [];

        for d = 1:numel(uniqueDays)
            rowsThisDay = find(dayIdx == d);

            % Ensure exact samplesPerDay samples for this day
            if numel(rowsThisDay) ~= samplesPerDay
                continue;
            end

            % Sort by time (just in case) and take the values in order
            [~, sortIdx] = sort(t_sel(rowsThisDay));
            dayVals = y_sel(rowsThisDay(sortIdx)).';

            data_matrix_year = [data_matrix_year; dayVals];
        end

        % Append to global matrix and record number of days 
        total_data_matrix           = [total_data_matrix; data_matrix_year]; % [totalDays x samplesPerDay] , each row = one weekday
        days_per_year(year_ind, 1)  = size(data_matrix_year, 1);
    end
end


%% Train Test Validation Split of dataset

function TrainValTestSplit(FileName, Total_data_matrix, Days_per_year)

    % Compute per-year day indices
    idxStart = cumsum([1; Days_per_year(1:end-1)]);
    idxEnd   = cumsum(Days_per_year);

    % Define train/val/test splits
    % 2015–2020 -> train
    % 2021–2023 -> validation
    % 2024      -> test
    train_idx = idxStart(1):idxEnd(6);
    val_idx   = idxStart(7):idxEnd(9);
    test_idx  = idxStart(10):idxEnd(10);

    % Extract days
    train_days = Total_data_matrix(train_idx, :);
    val_days   = Total_data_matrix(val_idx, :);
    test_days  = Total_data_matrix(test_idx, :);

    % Shuffle training days
    nTrainDays = size(train_days, 1);
    shuffled_order = randperm(nTrainDays);
    train_days = train_days(shuffled_order, :);




    % Build (X,Y) pairs for day-ahead forecasting 

    %  Training pairs 
    nTrainDays = size(train_days, 1);
    XTrain = cell(max(nTrainDays-1,0), 1);
    YTrain = cell(max(nTrainDays-1,0), 1);

    for k = 1:(nTrainDays-1)
        XTrain{k} = train_days(k,:); %  1 x samplesPerDay
        YTrain{k} = train_days(k+1,:); % 1 x samplesPerDay
    end

    % Validation pairs 
    nValDays = size(val_days, 1);
    XValidation = cell(max(nValDays-1,0), 1);
    YValidation = cell(max(nValDays-1,0), 1);

    for k = 1:(nValDays-1)
        XValidation{k} = val_days(k,:);
        YValidation{k} = val_days(k+1,:);
    end

    % Test pairs 
    nTestDays = size(test_days, 1);
    XTest = cell(max(nTestDays-1,0), 1);
    YTest = cell(max(nTestDays-1,0), 1);

    for k = 1:(nTestDays-1)   
        XTest{k} = test_days(k, :);
        YTest{k} = test_days(k+1, :);
    end


 
    save(FileName, ...
        "XTrain", "YTrain", "XValidation", "YValidation", "XTest", "YTest");

end



%% shortest path


function [ty, y] = shortest_path(t,low,high)
% Finds the shortest path bounded two curves,
% using the Dijkstra algorithm.
%
% Written by Y. Levron, Technion
%
% INPUTS:
% t    - time values, row vector
% low  - lower bound, row vector
% high - is the upper bound, row vector
%
% OUTPUTS
% ty   - time values
% y    - shortest path

% check for data integrity and running time
N = length(t);
if ((size(t,1)~=1)||(size(t,2)~=N)|| ...
    (size(low,1)~=1)||(size(low,2)~=N)|| ...   
    (size(high,1)~=1)||(size(high,2)~=N))
        'Error - all vectors must be row vectors of the same length';
        return
end
ind = find(low>high, 1);
if ~isempty(ind)
    'Error - low bound larger than upper bound';
    return;
end
if (N>1000)
    'Error - expected running time is too long';
    return
end
%%%%%%%%  end - data integrity   %%%%%%%%%%%%%%

very_small_number = ...
    0.001*min( [abs(low(2:end)-low(1:(end-1))) abs(high(2:end)-high(1:(end-1)))]);
% 
% mindt = abs(    t(2:end) - t(1:(end-1))   );
% 
% if (mindt < very_small_number)
%     'Error - problem is purely scaled.';
%     return
% end

% construct the distant matrix, between each pair of points.
dist2 = (-1)*ones(2*N);
for uu = 1:(2*N)

    if (uu<=N)
        t1 = t(uu);
        y1 = low(uu);
    else
        t1 = t(uu-N);
        y1 = high(uu-N);
    end

    for vv = uu:(2*N)
        if (uu==vv)
            dist2(uu,vv) = 0;
            continue;
        end

        if (vv<=N)
            t2 = t(vv);
            y2 = low(vv);
        else
            t2 = t(vv-N);
            y2 = high(vv-N);
        end

        % find the line connecting (t1,y1) -> (t2,y2)
        if (t1<=t2)
            ind_line = find((t1<=t).*(t<=t2));
        else
            ind_line = find((t2<=t).*(t<=t1));
        end
        
        if (t1==t2)
            dist2(uu,vv) = abs(y2-y1);
        else
            t_line = t(ind_line);
            a_line = (y2-y1)/(t2-t1);

            b_line = y1 - a_line*t1;
            y_line = a_line*t_line + b_line;

            ind = find((y_line>(high(ind_line)+very_small_number))+(y_line<(low(ind_line)-very_small_number)  ), 1);
            if (isempty(ind))
                dist2(uu,vv) = ((t2-t1)^2 + (y2-y1)^2)^0.5;
            else
                dist2(uu,vv) = inf;
            end
        end
    end
end

% duplicate the other side of the matrix:
for vv = 1:(2*N)
    for uu = vv:(2*N)
        dist2(uu,vv) = dist2(vv,uu);
    end
end
        
%%%%%%%%%% end matrix construction %%%%%%%%%%%%%%


%%%%%%%  run the Dijkstra algorithm  %%%%%%%
dist = inf*ones(1,2*N);
previous = NaN*ones(1,2*N);
dist(1) = 0;
Q = 1:(2*N);

while (~isempty(Q))
    min_dist_Q = min(dist(Q));
    u_Q = find(dist(Q)==min_dist_Q); u_Q=u_Q(1);
    u = Q(u_Q);
    
    if (isinf(dist(u)))
        'Error - destination is unreachable';
        disp(u);
        ty = NaN; y = NaN;
        return;
    end
    Q = Q(Q~=u);
    if (u == N)
        break;
    end
    
    alt = dist(u) + dist2(u,Q);
    ii = find(alt<dist(Q));  
    dist(Q(ii)) = alt(ii);
    previous(Q(ii)) = u;
end
%%%%%%% end Dijkstra algorithm %%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Build the shorest path %%%%%%%%%%
s = [];
u = N;
while (~isnan(u)) 
    s = [u s];
    u = previous(u);
end
%%%%%%% end shorest path %%%%%%%%%%

%%%%%%% translate vertixes to function  %%%%%%%%
ty = [];
y = [];
for ii = 1:length(s)
    u = s(ii);
    
    if (u<=N)
        t1 = t(u);
        y1 = low(u);
    else
        t1 = t(u-N);
        y1 = high(u-N);
    end
    
    ty = [ty t1];
    y = [y y1];
end
%%%%%%% end translate  %%%%%%%%%%%

return
end


%% Adding the optimization layer - Applying the Shortest Path on load profile 

function [Pg_0, E_max_storage, Ps_0, t] = Shortest_on_Load_and_plot(LoadProfile, E_max, print, Maintitle, SubTitle, legend1, legend2)

arguments
    LoadProfile
    E_max
    print = 0
    Maintitle = []
    SubTitle = 'The Optimal Generated Power P_g(t) and the Load Power P_L(t)'
    legend1 = 'Optimal Generated Power P_g'
    legend2 = 'Load Power P_L'
end

%  Time and scaling 
dt = 0.25;                 % hr
T = 24;
t = (T - dt*(length(LoadProfile)-1)):dt:T;  % ensure consistent length
t = t(:)';                 % make sure it's a row vector



%  Energy (MWh) and limits 
Load_in_MWH = cumtrapz(t, LoadProfile);
Load_in_MWH_max = Load_in_MWH + E_max;

%  Apply shortest path solver 
[ty_0, optimal_Eg_0] = shortest_path(t, Load_in_MWH, Load_in_MWH_max);

% Make sure ty_0 is strictly increasing with no duplicates
[ty_0_unique, idxUnique] = unique(ty_0, 'stable'); % emoves duplicate time points but keeps the original order.

% Apply the same selection to the corresponding values
optimal_Eg_0_unique = optimal_Eg_0(idxUnique);

% Now interpolate safely
optimal_Eg_0 = interp1(ty_0_unique, optimal_Eg_0_unique, t, 'pchip');

% Compute stored energy and powers 
optimal_E_stored_0 = optimal_Eg_0 - Load_in_MWH;
E_max_storage = max(optimal_E_stored_0);

% Derivatives (diff reduces length by 1, so re-interpolate)
Pg_mid = diff(optimal_Eg_0) ./ diff(t);
Ps_mid = diff(optimal_E_stored_0) ./ diff(t);

% interpolate back to same length as t
td = (t(2:end) + t(1:end-1)) / 2;
Pg_0 = interp1(td, Pg_mid, t, 'pchip', 'extrap');
Ps_0 = interp1(td, Ps_mid, t, 'pchip', 'extrap');

% ensure column vectors 
Pg_0 = Pg_0(:);
Ps_0 = Ps_0(:);
LoadProfile = LoadProfile(:);
t = t(:);

% Optional plotting 
if print
    figure;
    sgtitle(Maintitle,'FontSize',20,'FontWeight','bold');

    subplot(3,1,1);
    plot(t, optimal_Eg_0, 'g', 'LineWidth', 2); hold on;
    plot(t, Load_in_MWH, 'b', 'LineWidth', 1);hold on;
    plot(t, Load_in_MWH_max, 'm', 'LineWidth', 1);
    xlim([4 24]);
    title('The Optimal Generated Energy and Energy Bands','FontSize',15);
    xlabel('Time [Hour]','FontSize',12);
    ylabel('Energy [MWH]','FontSize',12);
    legend('W_g', 'W_L','W_L + W_{max}', 'Location', 'best');
    grid minor;

    subplot(3,1,2);
    plot(t, optimal_E_stored_0, 'm', 'LineWidth', 1.5);
    xlim([4 24]);
    title('The Optimal Stored Energy','FontSize',15);
    xlabel('Time [Hour]','FontSize',12);
    ylabel('Energy [MWH]','FontSize',12);
    legend('W_s', 'Location', 'best');
    grid minor;

    subplot(3,1,3);
    plot(t, Pg_0, 'm', 'LineWidth', 1.5); hold on;
    plot(t, LoadProfile, 'g', 'LineWidth', 1.5);
    xlim([4 24]);
    title(SubTitle, 'FontSize',15);
    xlabel('Time [Hour]','FontSize',12);
    ylabel('Power [W]','FontSize',12);
    legend('P_g', 'P_L', 'Location', 'best');
    grid minor;
end

end


%%
function [Pg_period, Emax_per_day] = Shortest_on_period(load_period, Emax, printFlag)

    if nargin < 3
        printFlag = 0;
    end

    nDays        = numel(load_period);
    Pg_period    = cell(nDays,1);
    Emax_per_day = zeros(nDays,1);

    for idx = 1:nDays
        day = load_period{idx}; % 1×T vector
        [Pg_period{idx}, Emax_per_day(idx)] = ...
            Shortest_on_Load_and_plot(day, Emax, printFlag);
    end
end



%%

function penalty_one_day = penalty_one_day_fnc(SP_actual, SP_pred, alpha)

    Runder = 0;
    Rover  = 0;

    for t = 1:numel(SP_actual)

        delta = SP_actual(t) - SP_pred(t);   % Δ = φ(P_L) − φ(P̂_L)

        if delta > 0
            % under-generation case
            Runder = Runder + abs(delta);
        elseif delta < 0
            % over-generation case
            Rover  = Rover  + abs(delta);
        end
    end

    % Weighted penalty
    Rtot = alpha * Runder + (1 - alpha) * Rover;

    % Normalization by "true generation area"
    denom = sum(SP_actual);

    penalty_one_day = Rtot / denom;

end


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Using Functions


%% Germany summer dataset
data_load_DE = [ ...
    "raw_data\Germany_Data\2015_Load.csv"
    "raw_data\Germany_Data\2016_Load.csv"
    "raw_data\Germany_Data\2017_Load.csv"
    "raw_data\Germany_Data\2018_Load.csv"
    "raw_data\Germany_Data\2019_Load.csv"
    "raw_data\Germany_Data\2020_Load.csv"
    "raw_data\Germany_Data\2021_Load.csv"
    "raw_data\Germany_Data\2022_Load.csv"
    "raw_data\Germany_Data\2023_Load.csv"
    "raw_data\Germany_Data\2024_Load.csv"];


T = readtable("raw_data\Germany_Data\2015_Load.csv");
T.Properties.VariableNames

vars = T.Properties.VariableNames;
LoadIdxDE = contains(vars, "Actual", "IgnoreCase", true) & ...
      contains(vars, "Load", "IgnoreCase", true)   & ...
      contains(vars, "MW",   "IgnoreCase", true);

TimeIdxDE = contains(vars, "MTU",  "IgnoreCase", true) & ...
           (contains(vars, "CET",  "IgnoreCase", true) | ...
            contains(vars, "CEST", "IgnoreCase", true));


loadColName_DE = vars{LoadIdxDE};
timeColName_DE = vars{TimeIdxDE};    



samplesPerDay_DE = 96;
initial_month_DE = 6;   % June
final_month_DE   = 9;   % September


[Total_data_matrix_DE, Days_per_year_DE] = buildWeekdaySeasonMatrixFromTime( ...
    data_load_DE, timeColName_DE, loadColName_DE, ...
    samplesPerDay_DE, initial_month_DE, final_month_DE);



%%
TrainValTestSplit("TrainValidationTest", ...
                    Total_data_matrix_DE, Days_per_year_DE);


%% Italy winter dataset


data_load_IT = [ ...
    "raw_data\Italy_Data\2015_Load.csv"
    "raw_data\Italy_Data\2016_Load.csv"
    "raw_data\Italy_Data\2017_Load.csv"
    "raw_data\Italy_Data\2018_Load.csv"
    "raw_data\Italy_Data\2019_Load.csv"
    "raw_data\Italy_Data\2020_Load.csv"
    "raw_data\Italy_Data\2021_Load.csv"
    "raw_data\Italy_Data\2022_Load.csv"
    "raw_data\Italy_Data\2023_Load.csv"
    "raw_data\Italy_Data\2024_Load.csv"];


I = readtable("raw_data\Italy_Data\2015_Load.csv");
I.Properties.VariableNames

vars = I.Properties.VariableNames;
LoadIdxIT = contains(vars, "Actual", "IgnoreCase", true) & ...
      contains(vars, "Load", "IgnoreCase", true)   & ...
      contains(vars, "MW",   "IgnoreCase", true);

TimeIdxIT = contains(vars, "MTU",  "IgnoreCase", true) & ...
           (contains(vars, "CET",  "IgnoreCase", true) | ...
            contains(vars, "CEST", "IgnoreCase", true));



loadColName_IT = vars{LoadIdxIT};
timeColName_IT = vars{TimeIdxIT};    


samplesPerDay_IT  = 24;
initial_month_IT  = 1;   % Jan–Feb
final_month_IT    = 2;

[total_data_matrix_IT, days_per_year_IT] = buildWeekdaySeasonMatrixFromTime( ...
    data_load_IT, timeColName_IT, loadColName_IT, ...
    samplesPerDay_IT, initial_month_IT, final_month_IT);

%% First option - use the trained best network 
% Load Test
S = load("TrainValidationTest.mat");
XTest  = S.XTest;
YTest  = S.YTest;

% Folder where the experiment lives
expFolder = "C:\Users\lidor\OneDrive\Desktop\FISP\Results\Experiment1_Result12_20251119T003413";

% Load the summary to get the best trial name
T = load(fullfile(expFolder,"TrialSummary.mat"));
best = T.T_sorted(1,:);              % first row = best
bestTrialName = best.Trial{1};       % e.g. 'Trial_13'

% Load the trained network from that trial
out = load(fullfile(expFolder, bestTrialName, "output.mat"));
net_best = out.results.trainedNetwork;

% Predict on the test set with the *experiment* model
YPred = predict(net_best, XTest, MiniBatchSize=1);


% Compute Test MAE
allAbsErr = [];
for k = 1:numel(YTest)
    absErr = abs(YPred{k} - YTest{k});
    allAbsErr = [allAbsErr; absErr(:)];
end
testMAE = mean(allAbsErr);

fprintf("Test MAE with best-trial network = %.3f MW\n", testMAE);




%% Second Option - retrain using the best hyperparameters found from the experiment 

S = load("TrainValidationTest.mat");
XTest  = S.XTest;
YTest  = S.YTest;
XTrain = S.XTrain;
YTrain = S.YTrain;
XVal = S.XValidation;
YVal = S.YValidation;


% Load hyperparameter summary and extract the best trial
expFolder = "C:\Users\lidor\OneDrive\Desktop\FISP\Results\Experiment1_Result12_20251119T003413";


summaryFile = fullfile(expFolder, "TrialSummary.mat");

if ~isfile(summaryFile)
    error("TrialSummary.mat not found in: %s", expFolder);
end

T = load(summaryFile);

best = T.T_sorted(1,:);   % first row is best (sorted by ValidationMAE)

% Extract best hyperparameters
LSTMDepth      = round(best.LSTMDepth);
numHiddenUnits = round(best.NumHiddenUnits);
initLR         = best.InitialLR;
dropoutRate    = best.Dropout;

fprintf("\nBEST hyperparameters loaded:\n");
fprintf("   Depth = %d\n", LSTMDepth);
fprintf("   Hidden Units = %d\n", numHiddenUnits);
fprintf("   Initial LR = %.6f\n", initLR);
fprintf("   Dropout = %.3f\n\n", dropoutRate);

% Network dimensions
featureDimension = 1;
numResponses     = 1;

% Build architecture
layers = sequenceInputLayer(featureDimension);

for i = 1:LSTMDepth
    layers = [layers;
              lstmLayer(numHiddenUnits, OutputMode="sequence")];
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
    InitialLearnRate       = initLR, ...
    LearnRateDropFactor    = 0.2, ...
    LearnRateDropPeriod    = 15, ...
    ValidationData         = {XVal, YVal}, ... % purely to monitor, not tune
    ValidationFrequency    = 20, ...
    GradientThreshold      = 1, ...
    Shuffle                = "never", ...
    Verbose                = false, ...
    OutputFcn              = @FISP_epochProgress); % @(info) FISP_epochProgress(info,maxEpochs));

% Train final model
fprintf("Training FINAL Germany model...\n");
[net, info] = trainNetwork(XTrain, YTrain, layers, options);


YPred = predict(net, XTest, MiniBatchSize=1);

% Compute Test MAE
allAbsErr = [];
for k = 1:numel(YTest)
    absErr = abs(YPred{k} - YTest{k});
    allAbsErr = [allAbsErr; absErr(:)];
end
testMAE = mean(allAbsErr);

fprintf("Test MAE with best-trial network = %.3f MW\n", testMAE);
% save("Germany_FinalModel.mat", "net_best", "YPred", "testMAE");
%%
save("GermanySummerPredTest.mat", "YPred", "XTest", "YTest","testMAE");

%%
GermanyData = load("GermanySummerPredTest.mat");
XTestGermany  = GermanyData.XTest;
YTestGermany  = GermanyData.YTest;
YPredGermany = GermanyData.YPred;
testMAEGermany = GermanyData.testMAE;

%%
ItalyData = load("ItalyWinterPredTest.mat");
XTestItaly  = ItalyData.XTest;
YTestItaly  = ItalyData.YTest;
YPredItaly = ItalyData.YPred;
%testMAEGermany = GermanyData.testMAE;

%%
GermanyARMA = load("Germany_Final_ARMA.mat");
GermanyARMATest = GermanyARMA.YTest_ARMA;
GermanyARMATest = cellfun(@(x) x.', GermanyARMATest, 'UniformOutput', false);
GermanyARMAPred = GermanyARMA.YPred_ARMA;
GermanyARMAPred = cellfun(@(x) x.', GermanyARMAPred, 'UniformOutput', false);


%%
ItalyARMA = load("Italy_Final_ARMA.mat");
ItalyARMATest = ItalyARMA.YTest_ARMA;
ItalyARMATest = cellfun(@(x) x.', ItalyARMATest, 'UniformOutput', false);
ItalyARMAPred = ItalyARMA.YPred_ARMA;
ItalyARMAPred = cellfun(@(x) x.', ItalyARMAPred, 'UniformOutput', false);




%% Choosing the year we want to predict and the prediction (between Germany and Italy)


% YearToPredict = YTestGermany;
% YearPrediciotn = YPredGermany;

%YearToPredict = YTestItaly;
%YearPrediciotn = YPredItaly;

%YearToPredict = GermanyARMATest;
%YearPrediciotn = GermanyARMAPred;

YearToPredict = ItalyARMATest;
YearPrediciotn = ItalyARMAPred;

%% Define Storage Sizes


Energies_storage_vector = zeros(numel(YearToPredict), 1);
E_max_large = 1e12; % effectively infinite storage

for idx = 1:numel(YearToPredict)
    day = YearToPredict{idx};               % extract the daily vector
    [~, Energies_storage_vector(idx)] = ...
        Shortest_on_Load_and_plot(day, E_max_large);
end

E_max = max(Energies_storage_vector);

energy_sizes_percentage = [5 10 20 30 40 100];
Storage_size = zeros([]);

for i = 1:size(energy_sizes_percentage,2)
    Storage_size(i) = (energy_sizes_percentage(i)/100)*E_max;
end


%% Applying 'Shortest Path' (with different storage sizes) on Prediction

numSizes = numel(Storage_size);
SP_LSTM = struct();

for i = 1:numSizes
    Ecap = Storage_size(i);

    label = sprintf("E_%dpercent", energy_sizes_percentage(i));

    fprintf("Running SP for storage = %d%% ...\n", energy_sizes_percentage(i));

    [Pg_period, Ws_day_max] = Shortest_on_period(YearPrediciotn, Ecap, 0);

    SP_LSTM.(label).Pg = Pg_period;
    SP_LSTM.(label).Ws_max = Ws_day_max;
    SP_LSTM.(label).Ecap = Ecap;
end





%% Applying 'Shortest Path' (with different storage sizes) on Actual Load

numSizes = numel(Storage_size);
SP_Actual = struct();

for i = 1:numSizes
    Ecap = Storage_size(i);

    label = sprintf("E_%dpercent", energy_sizes_percentage(i));

    fprintf("Running SP for storage = %d%% ...\n", energy_sizes_percentage(i));

    [Pg_period, Ws_day_max] = Shortest_on_period(YearToPredict, Ecap, 0);

    SP_Actual.(label).Pg = Pg_period;
    SP_Actual.(label).Ws_max = Ws_day_max;
    SP_Actual.(label).Ecap = Ecap;
end




%%

EmaxItaly = E_max;
StorageSizeItaly = Storage_size;
SP_LSTM_Italy = SP_LSTM;
SP_Actual_Italy = SP_Actual;

%% 

EmaxGermany = E_max;
StorageSizeGermany = Storage_size;
SP_LSTM_Germany = SP_LSTM;
SP_Actual_Germany = SP_Actual;

%%
EmaxGermanyARMA = E_max;
StorageSizeGermanyARMA = Storage_size;
SP_ARMA_Germany = SP_LSTM;
SP_ActualARMA_Germany = SP_Actual;

%%

EmaxItalyARMA = E_max;
StorageSizeItalyARMA = Storage_size;
SP_ARMA_Italy = SP_LSTM;
SP_ActualARMA_Italy = SP_Actual;




%%

SP_Actual = SP_ActualARMA_Italy;
SP_Pred   = SP_ARMA_Italy;
Storage_size = StorageSizeItalyARMA;
%%
numSizes      = numel(Storage_size);
mean_penalty  = struct();
alpha         = 0.7;   % chosen weighting

for i = 1:numSizes
    label = sprintf("E_%dpercent", energy_sizes_percentage(i));

    fprintf("Calculating Penalty for storage = %d%% ...\n", ...
            energy_sizes_percentage(i));

    SP_actual_all_days = SP_Actual.(label).Pg;   % cell array {nDays x 1}
    SP_pred_all_days   = SP_Pred.(label).Pg;     % cell array {nDays x 1}

    nDays     = numel(SP_actual_all_days);
    penalties = zeros(nDays, 1);

    for k = 1:nDays
        penalties(k) = penalty_one_day_fnc( ...
            SP_actual_all_days{k}, ...
            SP_pred_all_days{k}, ...
            alpha);
    end

    mean_penalty.(label).mean  = mean(penalties); % store the mean penalty for this storage size
    mean_penalty.(label).daily = penalties; 

    
end

PenaltItalyARMA = mean_penalty;


%% Creating Important Plots


%% Plot for Example LSTM Prediction Vs. Actual Load - Germany

samplesPerDay_DE = 96;
idx = 75;

t = datetime(2024,1,1) + minutes(15)*(0:(samplesPerDay_DE-1));

figure;
plot(t, YTestGermany{idx}, "b", "LineWidth", 2); hold on;
plot(t, YPredGermany{idx}, "m--", "LineWidth", 2);

title('Actual Load vs. LSTM Forecast - Germany Summer','FontSize',14,'FontWeight','bold');
legend('Actual Load','LSTM Forecast');
xlabel('Time','FontSize',12,'FontWeight','bold');
ylabel('Load (MW)','FontSize',12,'FontWeight','bold');

datetick('x','HH:MM','keeplimits');

grid on;

%% Plot for Example LSTM Prediction Vs. Actual Load - Italy

samplesPerDay = 24;
idx = 12;

t = datetime(2024,1,1) + minutes(60)*(0:(samplesPerDay-1));

figure;
plot(t, YTestItaly{idx}, "b", "LineWidth", 2); hold on;
plot(t, YPredItaly{idx}, "m--", "LineWidth", 2);

title('Actual Load vs. LSTM Forecast - Italy Winter','FontSize',14,'FontWeight','bold');
legend('Actual Load','LSTM Forecast');
xlabel('Time','FontSize',12,'FontWeight','bold');
ylabel('Load (MW)','FontSize',12,'FontWeight','bold');

datetick('x','HH:MM','keeplimits');

grid on;




%% Plot for Example SP on LSTM Prediction Vs. SP on Actual Load - Germany

capacityLabels = "E_" + string(energy_sizes_percentage) + "percent";

samplesPerDay_DE = 96;
idx = 75;   % which test day to show

% time axis for one day
t = datetime(2024,1,1) + minutes(15)*(0:(samplesPerDay_DE-1));


capacity_idx   = 5;  % 3 -> 'E_20percent'

spL = SP_LSTM_Germany.(capacityLabels{capacity_idx});   % LSTM SP
spA = SP_Actual_Germany.(capacityLabels{capacity_idx}); % Actual SP

Pg_LSTM   = spL.Pg{idx};
Pg_Actual = spA.Pg{idx};

figure;
plot(t, Pg_Actual, "b",    "LineWidth", 2); hold on;
plot(t, Pg_LSTM,   "m--", "LineWidth", 2);

% xtickformat('HH:mm')

title('Shortest-Path Generation: Actual vs. LSTM Forecast - Germany Summer', ...
      'FontSize',14,'FontWeight','bold');
legend('SP on Actual Load','SP on LSTM Forecast','Location','best');


xlabel('Time','FontSize',12,'FontWeight','bold');
ylabel('Generated Power (MW)','FontSize',12,'FontWeight','bold');

datetick('x','HH:MM','keeplimits');

grid on;




%% Plot for Example SP on LSTM Prediction Vs. SP on Actual Load - Italy

capacityLabels = "E_" + string(energy_sizes_percentage) + "percent";

samplesPerDay = 24;
idx = 12;   % which test day to show

% time axis for one day
t = datetime(2024,1,1) + minutes(60)*(0:(samplesPerDay-1));

capacity_idx   = 5;  % 3 -> 'E_20percent'

spL = SP_LSTM_Italy.(capacityLabels{capacity_idx});   % LSTM SP
spA = SP_Actual_Italy.(capacityLabels{capacity_idx}); % Actual SP

Pg_LSTM   = spL.Pg{idx};
Pg_Actual = spA.Pg{idx};

figure;
plot(t, Pg_Actual, "b",    "LineWidth", 2); hold on;
plot(t, Pg_LSTM,   "m--", "LineWidth", 2);

title('Shortest-Path Generation: Actual vs. LSTM Forecast - Italy Winter', ...
      'FontSize',14,'FontWeight','bold');
legend('SP on Actual Load','SP on LSTM Forecast','Location','best');

xlabel('Time','FontSize',12,'FontWeight','bold');
ylabel('Generated Power (MW)','FontSize',12,'FontWeight','bold');

datetick('x','HH:MM','keeplimits');

grid on;


%% Plot for Example ARMA Prediction Vs. Actual Load - Italy

samplesPerDay = 24;
idx = 12;

t = datetime(2024,1,1) + minutes(60)*(0:samplesPerDay-1);


figure;
plot(t, ItalyARMATest{idx}, 'b', 'LineWidth', 2); hold on;
plot(t, ItalyARMAPred{idx}, 'm--', 'LineWidth', 2);

title('Actual vs. ARMA Forecast – Italy winter');
xlabel('Time'); 
ylabel('Load (MW)');
legend('Actual','ARMA Forecast','Location','best');

datetick('x','HH:MM','keeplimits');

grid on;


%% Plot for Example ARMA Prediction Vs. Actual Load - Germany

samplesPerDay = 96;
idx = 76;

t = datetime(2024,1,1) + minutes(15)*(0:samplesPerDay-1);


figure;
plot(t, GermanyARMATest{idx}, 'b', 'LineWidth', 2); hold on;
plot(t, GermanyARMAPred{idx}, 'm--', 'LineWidth', 2);

title('Actual vs. ARMA Forecast – Germany summer');
xlabel('Time'); 
ylabel('Load (MW)');
legend('Actual','ARMA Forecast','Location','best');

datetick('x','HH:MM','keeplimits');

grid on;

%% Plot for Example SP on ARMA Prediction Vs. SP on Actual Load - Germany

capacityLabels = "E_" + string(energy_sizes_percentage) + "percent";

samplesPerDay_DE = 96;
idx = 75;   % which test day to show

% time axis for one day
t = datetime(2024,1,1) + minutes(15)*(0:(samplesPerDay_DE-1));


capacity_idx   = 5;  % 3 -> 'E_20percent'

spL = SP_ARMA_Germany.(capacityLabels{capacity_idx});   % LSTM SP
spA = SP_ActualARMA_Germany.(capacityLabels{capacity_idx}); % Actual SP

Pg_LSTM   = spL.Pg{idx};
Pg_Actual = spA.Pg{idx};

figure;
plot(t, Pg_Actual, "b",    "LineWidth", 2); hold on;
plot(t, Pg_LSTM,   "m--", "LineWidth", 2);


title('Shortest-Path Generation: Actual vs. ARMA Forecast - Germany Summer', ...
      'FontSize',14,'FontWeight','bold');
legend('SP on Actual Load','SP on ARMA Forecast','Location','best');


xlabel('Time','FontSize',12,'FontWeight','bold');
ylabel('Generated Power (MW)','FontSize',12,'FontWeight','bold');

datetick('x','HH:MM','keeplimits');

grid on;


%% Plot for Example SP on LSTM Prediction Vs. SP on Actual Load - Italy

capacityLabels = "E_" + string(energy_sizes_percentage) + "percent";

samplesPerDay = 24;
idx = 12;   % which test day to show

% time axis for one day
t = datetime(2024,1,1) + minutes(60)*(0:(samplesPerDay-1));

capacity_idx   = 5;  % 3 -> 'E_20percent'

spL = SP_ARMA_Italy.(capacityLabels{capacity_idx});   % LSTM SP
spA = SP_ActualARMA_Italy.(capacityLabels{capacity_idx}); % Actual SP

Pg_LSTM   = spL.Pg{idx};
Pg_Actual = spA.Pg{idx};

figure;
plot(t, Pg_Actual, "b",    "LineWidth", 2); hold on;
plot(t, Pg_LSTM,   "m--", "LineWidth", 2);

title('Shortest-Path Generation: Actual vs. ARMA Forecast - Italy Winter', ...
      'FontSize',14,'FontWeight','bold');
legend('SP on Actual Load','SP on ARMA Forecast','Location','best');

xlabel('Time','FontSize',12,'FontWeight','bold');
ylabel('Generated Power (MW)','FontSize',12,'FontWeight','bold');

datetick('x','HH:MM','keeplimits');

grid on;




%% Plot for Example LSTM Prediction Vs. Actual Load Vs. ARMA Prediction - Italy

samplesPerDay = 24;
idx = 12;

t = datetime(2024,1,1) + minutes(60)*(0:(samplesPerDay-1));

figure;
plot(t, YTestItaly{idx}, "g", "LineWidth", 2); hold on;
plot(t, YPredItaly{idx}, "m--", "LineWidth", 2); hold on;
plot(t, ItalyARMAPred{idx+1}, "b--", "LineWidth", 2);

title('Actual Load vs. LSTM Forecast vs. ARMA Forecast - Italy Winter','FontSize',14,'FontWeight','bold');
legend('Actual Load','LSTM Forecast','ARMA Forecast');
xlabel('Time','FontSize',12,'FontWeight','bold');
ylabel('Load (MW)','FontSize',12,'FontWeight','bold');

datetick('x','HH:MM','keeplimits');

grid on;


%% Plot for Example LSTM Prediction Vs. Actual Load Vs. ARMA Prediction - Germany

samplesPerDay_DE = 96;
idx = 75;

t = datetime(2024,1,1) + minutes(15)*(0:(samplesPerDay_DE-1));

figure;
plot(t, YTestGermany{idx}, "g", "LineWidth", 2); hold on;
plot(t, YPredGermany{idx}, "m--", "LineWidth", 2);hold on;
plot(t, GermanyARMAPred{idx+1}, "b--", "LineWidth", 2);


title('Actual Load vs. LSTM Forecast vs. ARMA Forecast - Germany Summer','FontSize',14,'FontWeight','bold');
legend('Actual Load','LSTM Forecast','ARMA Forecast');
xlabel('Time','FontSize',12,'FontWeight','bold');
ylabel('Load (MW)','FontSize',12,'FontWeight','bold');

datetick('x','HH:MM','keeplimits');

grid on;


%% ++ Plot for Example LSTM Prediction Vs. Actual Load Vs. ARMA Prediction - Germany


samplesPerDay_DE = 96;
idx = 75;

t = datetime(2024,1,1) + minutes(15)*(0:(samplesPerDay_DE-1));

f = figure('Color','w'); 
tiledlayout(f,1,1,'TileSpacing','compact','Padding','compact');

%  Storage power 
nexttile; hold on; box on; grid on;

p1 = plot(t, YTestGermany{idx}, "g", "LineWidth", 2);
p2 = plot(t, YPredGermany{idx}, "m--", "LineWidth", 2);
p3 = plot(t, GermanyARMAPred{idx+1}, "b--", "LineWidth", 2);
xlabel('Time [h]'); ylabel('Load P_L(t) [W]');
title('German Summer Day Load Profile (Actual vs. LSTM forecast vs. ARMA forecast)');
legend([p1 p2 p3], {'Actual-load','LSTM-forecast','ARMA-forecast'}, 'Location','best');
xlim([t(1) t(end)]);


set(findall(f,'-property','FontName'),'FontName','Times');
set(findall(f,'-property','FontSize'),'FontSize',20);




%% ++ Plot for Example LSTM Prediction Vs. Actual Load Vs. ARMA Prediction - Italy


samplesPerDay = 24;
idx = 12;

t = datetime(2024,1,1) + minutes(60)*(0:(samplesPerDay-1));

f = figure('Color','w'); 
tiledlayout(f,1,1,'TileSpacing','compact','Padding','compact');

%  Storage power 
nexttile; hold on; box on; grid on;

p1 = plot(t, YTestItaly{idx}, "g", "LineWidth", 2);
p2 = plot(t, YPredItaly{idx}, "m--", "LineWidth", 2);
p3 = plot(t, ItalyARMAPred{idx+1}, "b--", "LineWidth", 2);
xlabel('Time [h]'); ylabel('Load P_L(t) [W]');
title('Italian Winter Day Load Profile (Actual vs. LSTM forecast vs. ARMA forecast)');
legend([p1 p2 p3], {'Actual-load','LSTM-forecast','ARMA-forecast'}, 'Location','best');
xlim([t(1) t(end)]);


set(findall(f,'-property','FontName'),'FontName','Times');
set(findall(f,'-property','FontSize'),'FontSize',20);



%% Plot SP on LSTM vs, Actual - Italy

capacityLabels = "E_" + string(energy_sizes_percentage) + "percent";

samplesPerDay = 24;
idx = 12;   % which test day to show

% time axis for one day
t = datetime(2024,1,1) + minutes(60)*(0:(samplesPerDay-1));

capacity_idx   = 5;  % 3 -> 'E_20percent'

spLSTM = SP_LSTM_Italy.(capacityLabels{capacity_idx});   % LSTM SP
spAct = SP_Actual_Italy.(capacityLabels{capacity_idx}); % Actual SP

Pg_PredLSTM   = spLSTM.Pg{idx};
Pg_Actual = spAct.Pg{idx};


f = figure('Color','w'); 
tiledlayout(f,1,1,'TileSpacing','compact','Padding','compact');

%  Storage power 
nexttile; hold on; box on; grid on;

p1 = plot(t, Pg_Actual, "g", "LineWidth", 2);
p2 = plot(t, Pg_PredLSTM, "m--", "LineWidth", 2);
xlabel('Time [h]'); ylabel('Optimal Generation Power P_g(t) [W]');
title('Shortest Path Optimal Generation P_g (Actual vs. LSTM forecast) - Italian Winter ');
legend([p1 p2], {'SP on Actual-load','SP on LSTM-forecast'}, 'Location','best');
xlim([t(1) t(end)]);


set(findall(f,'-property','FontName'),'FontName','Times');
set(findall(f,'-property','FontSize'),'FontSize',12);


%%

print = 0;
idx = 75;
E_max = StorageSizeGermany(5);
[PgGermany, Emax, Ps_Actual, t] = Shortest_on_Load_and_plot(YTestGermany{idx}, E_max, print);
[PgGermanyLSRM, EMax, Ps_PredLSTM, t] = Shortest_on_Load_and_plot(YPredGermany{idx}, E_max, print);


%% Plot SP on LSTM vs, Actual - Germany

capacityLabels = "E_" + string(energy_sizes_percentage) + "percent";

samplesPerDay = 96;
idx = 75;   % which test day to show

% time axis for one day
t = datetime(2024,1,1) + minutes(15)*(0:(samplesPerDay-1));

capacity_idx   = 5;  % 3 -> 'E_20percent'

spLSTM = SP_LSTM_Germany.(capacityLabels{capacity_idx});   % LSTM SP
spAct = SP_Actual_Germany.(capacityLabels{capacity_idx}); % Actual SP

Pg_PredLSTM   = spLSTM.Pg{idx};
Pg_Actual = spAct.Pg{idx};


f = figure('Color','w'); 
tiledlayout(f,2,1,'TileSpacing','compact','Padding','compact');

%  Storage power 
nexttile; hold on; box on; grid on;
g1 = plot(t, Ps_Actual, "b", "LineWidth", 2);
g2 = plot(t, Ps_PredLSTM, "m--", "LineWidth", 2);
xlabel('Time [h]'); ylabel(' Power [W]');
title('Storage Power P_s (Actual vs. LSTM forecast) - German Summer ');
legend([g1 g2], {'P_s from SP on Actual-load','P_s from SP on LSTM-forecast'}, 'Location','best');
xlim([t(1) t(end)]);

nexttile; hold on; box on; grid on;
p1 = plot(t, Pg_Actual, "b", "LineWidth", 2);
p2 = plot(t, Pg_PredLSTM, "m--", "LineWidth", 2);
p3 = plot(t, YTestGermany{idx}, "g", "LineWidth", 1.5);
xlabel('Time [h]'); ylabel(' Power [W]');
title('Generated Power P_g  (Actual vs. LSTM forecast) and Load P_L - German Summer ');
legend([p1 p2 p3], {'P_g from SP on Actual-load','P_g from SP on LSTM-forecast', 'P_L Actual Load'}, 'Location','best');
xlim([t(1) t(end)]);


set(findall(f,'-property','FontName'),'FontName','Times');
set(findall(f,'-property','FontSize'),'FontSize',18);



%%
print = 1;
idx = 75;
LoadProfile = YTestGermany{idx};
E_max = EmaxGermany;
[PgGermanyLSTM, Emax, PsGermanyLSTM, t] = Shortest_on_Load_and_plot(LoadProfile, E_max, print);





%%
idx = 75;
E_max = EmaxGermany;
LoadProfile = YTestGermany{idx};
print = 1;

[Pg, E_max_storage, Ps, t] = Shortest_on_Load_and_plot(LoadProfile, E_max, print);




%%  penalty illustration: SP on actual vs SP on LSTM
% Choose storage size and test day
samplesPerDay = 96;
idx           = 75;   % which test day to show
capacity_idx  = 5;    % e.g. 5 -> 'E_40percent'

capacityLabels = "E_" + string(energy_sizes_percentage) + "percent";

% Get structs for this storage size
spLSTM = SP_LSTM_Germany.(capacityLabels{capacity_idx});   % LSTM SP
spAct  = SP_Actual_Germany.(capacityLabels{capacity_idx}); % Actual SP

% Extract generation profiles (ensure row vectors)
Pg_LSTM   = spLSTM.Pg{idx}(:)'; 
Pg_Actual = spAct.Pg{idx}(:)';

% Time axis as datetime for one arbitrary day
t = datetime(2024,1,1) + minutes(15)*(0:(samplesPerDay-1));

figure;
hold on; grid on;

%  Plot both SP trajectories
plot(t, Pg_Actual, 'b',  'LineWidth', 2);
plot(t, Pg_LSTM,  'm--','LineWidth', 2);

%  Shade the absolute difference region
x_fill = [t, fliplr(t)];
y_fill = [Pg_Actual, fliplr(Pg_LSTM)];
hFill  = fill(x_fill, y_fill, 'r', ...
              'FaceAlpha', 0.15, ...
              'EdgeColor', 'none');
uistack(hFill, 'bottom');  % send shading behind the lines

% Cosmetics
title('Shortest-Path Generation: Actual vs. LSTM Forecast', ...
      'FontSize', 20, 'FontWeight', 'bold');
xlabel('Time', 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Generated Power P_g(t) [MW]', 'FontSize', 15, 'FontWeight', 'bold');

% For datetime axes, use xtickformat instead of datetick
xtickformat('HH:mm');

legend({'Absolute Deviation Region', ...
        'SP on Actual Load', ...
        'SP on LSTM Forecast'}, ...
        'Location', 'best');

hold off;





%%  penalty illustration: SP on actual vs SP on ARMA
% Choose storage size and test day
samplesPerDay = 96;
idx           = 75;   % which test day to show
capacity_idx  = 5;    % e.g. 5 -> 'E_40percent'

capacityLabels = "E_" + string(energy_sizes_percentage) + "percent";

% Get structs for this storage size
spARMA = SP_ARMA_Germany.(capacityLabels{capacity_idx});   % LSTM SP
spAct  = SP_ActualARMA_Germany.(capacityLabels{capacity_idx}); % Actual SP

% Extract generation profiles (ensure row vectors)
Pg_ARMA  = spARMA.Pg{idx}(:)'; 
Pg_Actual = spAct.Pg{idx}(:)';

% Time axis as datetime for one arbitrary day
t = datetime(2024,1,1) + minutes(15)*(0:(samplesPerDay-1));

figure;
hold on; grid on;

%  Plot both SP trajectories
plot(t, Pg_Actual, 'b',  'LineWidth', 2);
plot(t, Pg_ARMA,  'm--','LineWidth', 2);

%  Shade the absolute difference region
x_fill = [t, fliplr(t)];
y_fill = [Pg_Actual, fliplr(Pg_ARMA)];
hFill  = fill(x_fill, y_fill, 'r', ...
              'FaceAlpha', 0.15, ...
              'EdgeColor', 'none');
uistack(hFill, 'bottom');  % send shading behind the lines

% Cosmetics
title('Shortest-Path Generation: Actual vs. ARMA Forecast', ...
      'FontSize', 20, 'FontWeight', 'bold');
xlabel('Time', 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Generated Power P_g(t) [MW]', 'FontSize', 15, 'FontWeight', 'bold');

% For datetime axes, use xtickformat instead of datetick
xtickformat('HH:mm');

legend({'Absolute Deviation Region', ...
        'SP on Actual Load', ...
        'SP on ARMA Forecast'}, ...
        'Location', 'best');

hold off;




