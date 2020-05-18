function outputLabel = FaceRecognition1(trainPath, testPath)

    % Retrieve training images and labels
    dims = [75 75]; % for reducing image size for efficiency
    folderNames = dir(trainPath);
    numImgs = length(folderNames) - 2;
    trainImgSet = zeros(dims(1) * dims(1), numImgs);
    labelImgSet = folderNames(3:end, :); % the folder names are the labels

    for i = 3:length(folderNames)
        imgName = dir([trainPath, folderNames(i, :).name, '/*.jpg']).name;
        image = imread([trainPath, folderNames(i, :).name, '/', imgName]);
        image = imresize(image, dims);
        image = rgb2gray(image);
        image = reshape(image, [], 1);

        % each column is an image
        trainImgSet(:, i - 2) = image';
    end

    %%
    % find the average face
    meanFace = mean(trainImgSet, 2);
    % find deviation from the mean for each face
    devImages = trainImgSet - repmat(meanFace, 1, numImgs);

    coeff = pca(trainImgSet');

    coeff = coeff(:, 1:25); % reduce dimensionality
    descriptors = coeff' * devImages;

    %%
    testImgNames = dir([testPath, '*.jpg']);
    outputLabel = char(zeros(size(testImgNames, 1), 6));

    for i = 1:size(testImgNames, 1)
        testImg = imread([testPath, testImgNames(i, :).name]);
        testImg = imresize(testImg, dims);
        testImg = rgb2gray(testImg);
        testImg = reshape(testImg, [], 1);
        testImg = double(testImg);

        % vector that describes the image
        descriptor = coeff' * (testImg - meanFace);
        % find euclidean distance from each descriptor
        distance = arrayfun(@(x) norm(descriptors(:, x) - descriptor), 1:numImgs);

        % find the smallest distance
        [~, idx] = min(distance);
        outputLabel(i, :) = labelImgSet(idx, :).name; % store the outputLabels for each of the test image
    end

end
