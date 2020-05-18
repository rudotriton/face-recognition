clear all;
close all;
trainPath = './FaceDatabase/Train/'; % provide full path here
testPath = './FaceDatabase/Test/';
%% Baseline Method
tic;
outputLabel = FaceRecognition(trainPath, testPath);
baseLineTime = toc;

%% Eval
load testLabel
correctP = 0;

for i = 1:size(testLabel, 1)

    if outputLabel(i, :) == testLabel(i, :)
        correctP = correctP + 1;
    end

end

recAccuracy = correctP / size(testLabel, 1) * 100; %Recognition accuracy%

%% Eigenfaces
tic;
outputLabel1 = FaceRecognition1(trainPath, testPath);
method1Time = toc;

% Eval
load testLabel
correctP = 0;

for i = 1:size(testLabel, 1)

    if outputLabel1(i, :) == testLabel(i, :)
        correctP = correctP + 1;
    end

end

recAccuracy1 = correctP / size(testLabel, 1) * 100; %Recognition accuracy%

%% CNN

% Matlab does not recognize the complete sourced system path,
% which requires setting the NodeJS path before executing a script with it.
setenv('PATH', [getenv('PATH') ':/Users/raigo/.nvm/versions/node/v13.5.0/bin']);

tic;
outputLabel2 = FaceRecognition2(trainPath, testPath);
method2Time = toc;

% Eval
load testLabel
correctP = 0;

for i = 1:size(testLabel, 1)

    if outputLabel2(i, :) == testLabel(i, :)
        correctP = correctP + 1;
    end

end

recAccuracy2 = correctP / size(testLabel, 1) * 100; %Recognition accuracy%
