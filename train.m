function validationAccuracies = train(trainData, trainLabels, validData, validLabels, k)
    figure;
    subplot(2,1,1)
    hValPlot = animatedline('Color',[0.8500 0.3250 0.0980]	,'Marker', 'o', 'LineWidth', 2, 'DisplayName', 'Validation Accuracy');
    xlabel('Itteration');
    ylabel('Accuracy');
    title('Learning Graph');
    legend('Location', 'Best');
    grid on;
    subplot(2,1,2);
    bar(0, 'FaceColor', [0.5, 0.5, 0.5]);
    xlabel('Itterations');
    ylabel('Accuracy Value');
    title('Learning Bars');
    % Initialize arrays to store validation accuracies
    validationAccuracies = zeros(1, k);
    
    % Determine fold size (assuming the validation set is already defined)
    foldSize = floor(numel(trainLabels) / k);
    
    % Perform k-fold cross-validation
    for i = 1:k
        % Extract training data for current fold (excluding validation set)
        startIdx = (i - 1) * foldSize + 1;
        endIdx = min(i * foldSize, numel(trainLabels));
        trainIndices = [1:startIdx-1, endIdx+1:numel(trainLabels)];
        foldTrainData = trainData(trainIndices, :);
        foldTrainLabels = trainLabels(trainIndices);
        
        % Train SVM model on current fold's training data
        svmModel = fitcsvm(foldTrainData, foldTrainLabels);
        
        % Evaluate model on validation data
        validPredictions = predict(svmModel, validData);
        validationAccuracies(i) = sum(validPredictions == validLabels) / numel(validLabels);

        % Update live plots
        subplot(2,1,1);
        addpoints(hValPlot, i, validationAccuracies(i));
        drawnow;
        subplot(2,1,2);
        bar(validationAccuracies, 'FaceColor', [0.5, 0.5, 0.5]);
        drawnow;
    end
    xlabel('Itterations');
    ylabel('Accuracy Value');
    title('Learning Bars');
    % Calculate median accuracy
    medianAccuracy = median(validationAccuracies);
    avAccuracy = mean(validationAccuracies);
    stdAccuracy = std(validationAccuracies);
    % Display median accuracy
    fOutput=fopen('Output.txt','w');
    fprintf(fOutput,'Median accuracy of SVM model: %.2f%%\n', medianAccuracy * 100);
    fprintf('Median accuracy of SVM model: %.2f%%\n', medianAccuracy * 100);
    fprintf(fOutput,'Average accuracy of SVM model: %.2f%%\n', avAccuracy * 100);
    fprintf('Average accuracy of SVM model: %.2f%%\n', avAccuracy * 100);
    fprintf(fOutput,'Standard Deviation accuracy of SVM model: %.2f%%\n', stdAccuracy * 100);
    fprintf('Standard Deviation accuracy of SVM model: %.2f%%\n', stdAccuracy * 100);
    fclose(fOutput);
    save svmModel.mat;
end
