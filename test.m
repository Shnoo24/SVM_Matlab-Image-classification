clear all;
close all;
clc;
%================================Testing Part=======================================
% Example of evaluating model performance (Better to use cross-validation or separate test set)
load svmModel.mat;
classes = ["Football","Basketball"];
image_size = [64, 64];
features_test = [];
while(1)
    try
        [fname, path,type]=uigetfile({'*.jpg';'*.jpeg';'*.png';'*.tif';'*.gif'},'Open an Image as input to Classify It','ImageName.jpg');
        if fname == 0
            return;
        end
        fname=strcat(path, fname);
        img = imread(fname);
        break;
    catch
            disp("This type is not Supported");
    end
end

img = imresize(img, image_size);
if size(img, 3) == 3
    img = rgb2gray(img);
end
hog_feature = extractHOGFeatures(img, 'CellSize', [4 4], 'BlockSize', [1 1], 'NumBins', 20);
features_test = [features_test; hog_feature];
Y_pred = predict(svmModel, features_test);
disp(['Class: ',char(classes(Y_pred+1))]);