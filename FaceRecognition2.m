function outputLabel = FaceRecognition2(trainPath, testPath)
    
    % hand-off to external script
    system(strjoin({'node', 'faceRecognition.js', trainPath, testPath}), '-echo');
    outputLabel = readmatrix('results.csv', 'OutputType','string');
end
