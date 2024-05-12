clear all;
close all;
clc;
try
    load svmModel.mat
catch

% Loading the Paths of training and validation set from csv file
paths_train = readtable('Paths_train.csv', 'Delimiter', ',');
paths_valid = readtable('Paths.csv', 'Delimiter', ',');
% Storing Football training and validation images Paths
images_Foot_train = paths_train.Football(~cellfun(@isempty, paths_train.Football));
images_Foot_valid = paths_valid.Football(~cellfun(@isempty, paths_valid.Football));
% Calculating  the number of Football images
num_images_Foot_train = length(images_Foot_train);
num_images_Foot_valid = length(images_Foot_valid);
% Storing Basketball training and validation images Paths
images_Basket_train = paths_train.Basketball(~cellfun(@isempty, paths_train.Basketball));
images_Basket_valid = paths_valid.Basketball(~cellfun(@isempty, paths_valid.Basketball));
% Calculating  the number of Basketball images
num_images_Basket_train = length(images_Basket_train);
num_images_Basket_valid = length(images_Basket_valid);
% Deciding the scale of resizing Images to 64x64 pixels
image_size = [64, 64];

%===============================================================================================
% Preprocessing images and extracting HOG features
% Intialize the Features arrays for each kind of balls for the splited Data
features_Foot_train = []; 
features_Foot_valid= [];
features_Basket_train = [];
features_Basket_valid = [];

loading_graphics = ['|','/','-','\']; % Character array to use it in visualization of loading text

% Define SPM parameters (e.g., number of levels, bins per level)
levels = 2; % Number of pyramid levels
bins = 9; % Number of histogram bins

%4 Loops to exctract The Features of each Image Dataset

for i = 1:num_images_Foot_train
    clc;
    disp("Loading Data: "+loading_graphics(modByConstant(i,4)+1))
    
    try
    img = imread(char(images_Foot_train(i)));
    catch
    continue;
    end
    
    img = imresize(img, image_size);
    % Convert to grayscale if Images are RGB
    if size(img, 3) == 3
       img = rgb2gray(img);
    end
    hog_feature = extractHOGFeatures(img, 'CellSize', [4 4], 'BlockSize', [1 1], 'NumBins', 20);
    features_Foot_train = [features_Foot_train; hog_feature];
end

for i = 1:num_images_Basket_train
    clc;
    disp("Loading Data: "+loading_graphics(modByConstant(i,4)+1)+" exceed 25% of process");
    
    try
    img = imread(char(images_Basket_train(i)));
    catch
    continue;
    end
    img = imresize(img, image_size);
    % Convert to grayscale if Images are RGB
    if size(img, 3) == 3
       img = rgb2gray(img);
    end
    hog_feature = extractHOGFeatures(img, 'CellSize', [4 4], 'BlockSize', [1 1], 'NumBins', 20);
    features_Basket_train = [features_Basket_train; hog_feature];
end

for i = 1:num_images_Foot_valid
    clc;
    disp("Loading Data: "+loading_graphics(modByConstant(i,4)+1)+" exceed 50% of process");
    
    try
    img = imread(char(images_Foot_valid(i)));
    catch
    continue;
    end
    img = imresize(img, image_size);
    % Convert to grayscale if Images are RGB
    if size(img, 3) == 3
       img = rgb2gray(img);
    end
    hog_feature = extractHOGFeatures(img, 'CellSize', [4 4], 'BlockSize', [1 1], 'NumBins', 20);
    features_Foot_valid = [features_Foot_valid; hog_feature];
end

for i = 1:num_images_Basket_valid
    clc;
    disp("Loading Data: "+loading_graphics(modByConstant(i,4)+1)+" exceed 75% of process");
    
    try
    img = imread(char(images_Basket_valid(i)));
    catch
    continue;
    end
    img = imresize(img, image_size);
    % Convert to grayscale if Images are RGB
    if size(img, 3) == 3
       img = rgb2gray(img);
    end
    hog_feature = extractHOGFeatures(img, 'CellSize', [4 4], 'BlockSize', [1 1], 'NumBins', 20);
    features_Basket_valid = [features_Basket_valid; hog_feature];
end
clc;

if(isempty(features_Foot_train) == 1 ||isempty(features_Basket_train) == 1)
    if(isempty(features_Basket_train) == 1)
        disp("Basket Training Dataset Corrupted");
        return;
    else
        disp("Football Training Dataset Corrupted");
        return;
    end

elseif(isempty(features_Foot_valid) == 1 || isempty(features_Basket_valid) == 1)
    if(isempty(features_Basket_valid) == 1)
        disp("Basket Validation Dataset Corrupted");
        return;
    else
        disp("Football Validation Dataset Corrupted");
        return;
    end

else
disp("All Data Loaded Successfully 100%.");
end

% Concatenate features and labels to pass it Into the training Function
trainData = [features_Foot_train; features_Basket_train];
trainLabels = [zeros(num_images_Foot_train, 1); ones(num_images_Basket_train, 1)]; % Labels: 0 for class Football, 1 for class Basketball
validData = [features_Foot_valid; features_Basket_valid];
validLabels = [zeros(num_images_Foot_valid, 1); ones(num_images_Basket_valid, 1)];
%Training the Network and Validate
accuracies=train(trainData, trainLabels, validData, validLabels, 10);
end
test_choice = input('\nDo you want to test the Nueral Network by yourself using Image from your computer (y/Yes or n/No): ','s');
if(test_choice(1) == 'y' || test_choice(1) == 'Y')
    test;
elseif(test_choice(1) == 'n' || test_choice(1) == 'N')
    disp("Ok have fun Whenever you want to test the Network Makesure You are in Code Folder and write `test` ");
    disp("Bye :(");
    return;
end