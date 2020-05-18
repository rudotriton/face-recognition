function outputLabel = eigenWDetect(trainPath, testPath)
    %% Face recognition using eigenfaces

    % Retrieve training images and labels
    folderNames = dir(trainPath);
    numImgs = length(folderNames) - 2;
    trainImgSet = zeros(600, 600, numImgs);
    labelImgSet = folderNames(3:end, :); % the folder names are the labels

    % set up a face detector
    detector = vision.CascadeObjectDetector;
    boundaries = zeros(numImgs, 4); % matrix for bounding boxes of found faces

    %%
    for i = 3:length(folderNames)
        imgName = dir([trainPath, folderNames(i, :).name, '/*.jpg']).name;
        image = imread([trainPath, folderNames(i, :).name, '/', imgName]);

        bb = step(detector, image);

        if isempty(bb)
            bb = [1 1 599 599];
        end

        % sometimes seemingly random spots are id'd as faces
        [~, idx] = max(bb(:, 3));
        boundaries(i - 2, :) = bb(idx, :);

        trainImgSet(:, :, i - 2) = rgb2gray(image); % store image
    end

    %% crop faces and resize images
    % dimensions based on the smallest patch
    %     dims = min(boundaries(:,3));
    dims = 100;
    croppedImgs = zeros(dims * dims, numImgs);

    for i = 1:numImgs
        croppedImg = cropFace(trainImgSet(:, :, i), boundaries(i, :));
        croppedImg = imresize(croppedImg, [dims, dims]);

        croppedImg = reshape(croppedImg, dims * dims, []);

        croppedImgs(:, i) = croppedImg;
    end

    %%
    % find the average face
    meanFace = mean(croppedImgs, 2);
    % find deviation from the mean for each face
    devImages = croppedImgs - repmat(meanFace, 1, numImgs);

    coeff = pca(croppedImgs');

    coeff = coeff(:, 1:25); % reduce dimensionality
    descriptors = coeff' * devImages;

    %%
    testImgNames = dir([testPath, '*.jpg']);
    outputLabel = char(zeros(size(testImgNames, 1), 6));

    for i = 1:size(testImgNames, 1)
        testImg = imread([testPath, testImgNames(i, :).name]);

        bb = step(detector, testImg);

        if isempty(bb)
            bb = [1 1 599 599];
        end

        [~, idx] = max(bb(:, 3));
        bb = bb(idx, :);

        croppedImg = cropFace(testImg, bb);
        croppedImg = imresize(croppedImg, [dims, dims]);

        croppedImg = reshape(croppedImg, dims * dims, []);
        croppedImg = double(croppedImg);

        % vector that describes the image
        descriptor = coeff' * (croppedImg - meanFace);
        % find euclidean distance from each descriptor
        distance = arrayfun(@(x) norm(descriptors(:, x) - descriptor), 1:numImgs);

        % find the smallest distance
        [~, idx] = min(distance);
        % show testImg and its match size by side
        % imshow([uint8(reshape(croppedImg, dims, [])), uint8(reshape(croppedImgs(:, idx), dims, []))])
        outputLabel(i, :) = labelImgSet(idx, :).name; % store the outputLabels for each of the test image
    end

end
