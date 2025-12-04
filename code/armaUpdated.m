

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

%%

function valMAE = computeValidationMAE_ARMA(mdlEst, seriesVal, samplesPerDay)

    nVal   = numel(seriesVal);
    nDays  = nVal / samplesPerDay;
    if mod(nVal, samplesPerDay) ~= 0
        error('Validation series length is not an integer multiple of samplesPerDay.');
    end

    allAbsErr = [];

    % We forecast day d using all validation data *up to the day before*
    for d = 1:nDays
        % history: everything before this day in the validation block
        idxHistEnd  = (d-1)*samplesPerDay;
        if idxHistEnd == 0
            % If there is no previous day in the validation block,
            % we use the first day as history to get the recursion started.
            yPast = seriesVal(1:samplesPerDay);
        else
            yPast = seriesVal(1:idxHistEnd);
        end

        % one-step-ahead 96-step forecast for this day
        yHat = forecast(mdlEst, samplesPerDay, 'Y0', yPast);

        % true values for this day
        idxDayStart = idxHistEnd + 1;
        idxDayEnd   = idxHistEnd + samplesPerDay;
        yTrue       = seriesVal(idxDayStart:idxDayEnd);

        allAbsErr = [allAbsErr; abs(yHat - yTrue)];
    end

    valMAE = mean(allAbsErr, 'omitnan');
end



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


[data_matrix_DE, days_per_year_DE] = buildWeekdaySeasonMatrixFromTime( ...
    data_load_DE, timeColName_DE, loadColName_DE, ...
    samplesPerDay_DE, initial_month_DE, final_month_DE);
%%

Days_per_year = days_per_year_IT; % [Ndays x samplesPerDay]
Total_data_matrix = total_data_matrix_IT; % vector length = 10 years
samplesPerDay = samplesPerDay_IT; % 96 for DE

% Sanity check
[Ndays, T] = size(Total_data_matrix);
fprintf("Total days = %d, samplesPerDay = %d\n", Ndays, T);

%%
% global time series (all days concatenated) Row-by-row: [day1 ; day2 ; ...]
seriesAll = reshape(Total_data_matrix.', [], 1);   % [Ndays * samplesPerDay  × 1]

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

fprintf("Train days  : %d\n", numel(train_idx));
fprintf("Val days    : %d\n", numel(val_idx));
fprintf("Test days   : %d\n", numel(test_idx));

% Extract days
%train_days = Total_data_matrix(train_idx, :);
%val_days   = Total_data_matrix(val_idx, :);
%test_days  = Total_data_matrix(test_idx, :);



dayToSamples = @(idx) reshape( ...
    ((idx(:)-1) * samplesPerDay + (1:samplesPerDay)).', [], 1);

idxTrainSamples = dayToSamples(train_idx);
idxValSamples   = dayToSamples(val_idx);
idxTestSamples  = dayToSamples(test_idx);

seriesTrain = seriesAll(idxTrainSamples);
seriesVal   = seriesAll(idxValSamples);
seriesTest  = seriesAll(idxTestSamples);

fprintf("Train samples = %d\n", numel(seriesTrain));
fprintf("Val   samples = %d\n", numel(seriesVal));
fprintf("Test  samples = %d\n", numel(seriesTest));




%% Search grid of the best ARMA order

candidateP = 0:8;
candidateQ = 0:8;

bestMAE = inf;
bestPQ  = [NaN NaN];

totalModels = numel(candidateP) * numel(candidateQ) - 1;  % minus the (0,0) we skip
modelCount  = 0;



for p = candidateP
    for q = candidateQ

         % Skip trivial ARMA(0,0), pure white noise, usually not interesting
        if p==0 && q==0
            continue;  
        end

        modelCount = modelCount + 1;
        fprintf("Trying ARMA(%d,%d) [%d / %d]...\n", ...
                p, q, modelCount, totalModels);
        try

            % Define model structure
            mdl = arima('Constant', NaN, ...
                        'ARLags', 1:p, ...
                        'MALags', 1:q);


            % Estimate parameters on training series
            mdlEst = estimate(mdl, seriesTrain, 'Display','off');

            
            % day-ahead predictions for the validation years
            % and compute MAE over all validation days:
            valMAE = computeValidationMAE_ARMA(mdlEst, seriesVal, samplesPerDay);
            fprintf("   → Validation MAE = %.4f\n", valMAE);

            % Check if this is the best so far
            if valMAE < bestMAE
                bestMAE = valMAE;
                bestPQ  = [p q];
                fprintf("New BEST so far: ARMA(%d,%d) with MAE = %.4f\n", ...
                        p, q, bestMAE);
            end

        catch ME
            % Some orders may fail to converge: just skip them
            warning("ARMA(%d,0,%d) failed: %s", p, q, ME.message);
        end
    end
end


fprintf("\n=== Final best ARMA order ===\n");
fprintf("p = %d, q = %d with validation MAE = %.4f\n\n", ...
        bestPQ(1), bestPQ(2), bestMAE);


%% Training and evaluating ARMA best model

%Load best ARMA order (p,q) from grid search 
% Here we assume bestPQ = [pBest, qBest] is already in workspace.

pBest = bestPQ(1);
qBest = bestPQ(2);

fprintf("\nUsing FINAL ARMA order: p = %d, q = %d\n", pBest, qBest);

% Train final ARMA on TRAIN+VAL 

% Concatenate train + val for final estimation
seriesTrainVal = [seriesTrain; seriesVal];


% Build ARMA(p,0,q) model
mdlFinal = arima('Constant', NaN, ...
                 'ARLags', 1:pBest, ...
                 'MALags', 1:qBest);

fprintf("Estimating FINAL ARMA(%d,%d) model on train+val...\n", pBest, qBest);
mdlFinalEst = estimate(mdlFinal, seriesTrainVal, 'Display','off');
fprintf("... estimation complete.\n");

%% Day-ahead forecasts for ALL test days (2024) 

nTestDays = numel(test_idx);


YPred_ARMA = cell(nTestDays, 1);
YTest_ARMA = cell(nTestDays, 1);

maxLag   = max(bestPQ);  

for k = 1:nTestDays
    d = test_idx(k);   % global day index within Total_data_matrix

    % Samples for this test day in global series
    daySampleIdx = (d-1)*samplesPerDay + (1:samplesPerDay);

    % Actual load for this day
    trueDay = seriesAll(daySampleIdx);
    YTest_ARMA{k} = trueDay;

    % History just before the test day
    startSample = daySampleIdx(1);   % first sample of that day
    histEnd     = startSample - 1;   % last sample before the day
    histStart   = histEnd - maxLag + 1;

    if histStart < 1
        error("Not enough history samples before test day %d for ARMA forecast.", d);
    end

    Y0 = seriesAll(histStart:histEnd);

    % forecast with that history:
    YF = forecast(mdlFinalEst, samplesPerDay, 'Y0', Y0);
    YPred_ARMA{k} = YF;

    fprintf("  Day %d/%d done.\n", k, nTestDays);
end

fprintf("... all ARMA test forecasts completed.\n");



%%  Compute Test MAE for ARMA (Italy, 2024)
allAbsErr = [];

for k = 1:nTestDays
    err = abs(YPred_ARMA{k} - YTest_ARMA{k});
    allAbsErr = [allAbsErr; err(:)];
end

testMAE_ARMA = mean(allAbsErr);
%%
fprintf("\nFINAL ARMA Test MAE (Italy, 2024) = %.3f MW\n", testMAE_ARMA);

%%  Save everything for later use in FISP 
save("Italy_Final_ARMA.mat", ...
     "mdlFinalEst", "pBest", "qBest", ...
     "YPred_ARMA", "YTest_ARMA", "testMAE_ARMA", ...
     "samplesPerDay", "test_idx");
%%
fprintf("Saved final ARMA results to Italy_Final_ARMA.mat\n");

%%
size(t)
size(YTest_ARMA{idx})
size(YPred_ARMA{idx})
%%
t_col     = t(:);
true_col  = YTest_ARMA{idx}(:);
pred_col  = YPred_ARMA{idx}(:);

%%

idx = 24;

t = datetime(2024,1,1) + minutes(15)*(0:samplesPerDay_DE-1);

% Ensure all are column vectors
t = t(:);
trueDay = YTest_ARMA{idx}(:);
predDay = YPred_ARMA{idx}(:);

figure;
plot(t, trueDay, 'b', 'LineWidth', 2); hold on;
plot(t, predDay, 'm--', 'LineWidth', 2);

title(sprintf('Actual vs. ARMA Forecast – Test Day %d', idx));
xlabel('Time'); 
ylabel('Load (MW)');
legend('Actual','ARMA Forecast','Location','best');
grid on;








