function stop = FISP_epochProgress(info)
%   Prints a single line whenever a new epoch starts.
%   info.State can be: "start", "iteration", "done"

persistent lastEpoch

stop = false;   % we never stop early in this function

switch info.State
    case "start"
        lastEpoch = 0;
        fprintf('\n[Training started]\n');

    case "iteration"
        % Print once per epoch (at the first iteration of that epoch)
        if info.Epoch > lastEpoch
            lastEpoch = info.Epoch;
            fprintf('  Epoch %3d  Iteration %4d  TrainLoss = %.4f', ...
                info.Epoch, info.Iteration, info.TrainingLoss);

            if isfield(info,'ValidationLoss') && ~isempty(info.ValidationLoss)
                fprintf('  ValLoss = %.4f', info.ValidationLoss);
            end
            fprintf('\n');
        end

    case "done"
        fprintf('[Training finished] Final TrainingLoss = %.4f\n', ...
            info.TrainingLoss);
end
end
