%% Collecting Results from experiment manager trial folders 

% Path to your experiment folder:
expFolder = "C:\Users\lidor\OneDrive\Desktop\FISP\Results\Experiment1_Result12_20251119T003413";

trials = dir(fullfile(expFolder, "Trial_*"));

trialNums = arrayfun(@(d) sscanf(d.name, 'Trial_%d'), trials);
[~, order] = sort(trialNums);
trials = trials(order);

% Prepare storage
resultsSummary = struct([]);
row = 1;

for k = 1:numel(trials)
    trialName = trials(k).name;
    trialPath = fullfile(expFolder, trialName);

    % Try loading input.mat (hyperparameters)
    inputFile = fullfile(trialPath, "input.mat");
    outputFile = fullfile(trialPath, "output.mat");

    if ~isfile(inputFile) || ~isfile(outputFile)
        warning("Missing input/output for %s", trialName);
        continue;
    end

    % Load both files
    in = load(inputFile);    % typically contains params struct
    out = load(outputFile);  % contains results struct

    try
        params = in.input.paramValues;        % Experiment Manager format
    catch
        params = in.paramValues;              % fallback format
    end

    if numel(params) > 1
        params = params(1);
    end

    try
        res = out.output.results;        % Experiment Manager format
    catch
        res = out.results;               % fallback format
    end

    % Store everything into our summary struct
    resultsSummary(row).Trial      = trialName;
    resultsSummary(row).LSTMDepth  = params.LSTMDepth;
    resultsSummary(row).NumHiddenUnits = params.NumHiddenUnits;
    resultsSummary(row).InitialLR  = params.InitialLearnRate;
    resultsSummary(row).Dropout    = params.DropoutRate;

    resultsSummary(row).ValidationMAE = res.ValidationMAE;

    row = row + 1;
end

%% Convert to table for easier viewing
T = struct2table(resultsSummary);

%% Sort by best (lowest) MAE
T_sorted = sortrows(T, "ValidationMAE");

%% Save results
save(fullfile(expFolder, "TrialSummary.mat"), "T", "T_sorted");
writetable(T_sorted, fullfile(expFolder, "TrialSummary.csv"));

%% Display best trial
disp("===== BEST TRIAL =====");
disp(T_sorted(1,:));
