function croppedImage = cropFace(image, boundingBox)
    % Crop and image to size 
    %   Given an image and boundingBox (x, y, width, height) crop the image
    % to the given dimensions from the x and y coords
    x = boundingBox(1);
    y = boundingBox(2);
    w = boundingBox(3);
    h = boundingBox(4);
    croppedImage = image(x:x + w, y:y + h);
end
