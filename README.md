# Face Recognition

> This project has been developed for a computer vision course and includes two different methods for face recognition.

## Requirements

- [NodeJS](https://nodejs.org/en/) - tested with [v13.5.0](https://nodejs.org/en/download/releases/)
  - [For when node-gyp fails on MacOS](https://github.com/nodejs/node-gyp/blob/master/macOS_Catalina.md)
- `MATLAB`
- `Python 2.7` is required by `@tensorflow/tfjs-node`

## Running

Note: the code may also need changes to be run on Windows machines.

- `npm install` - to download the necessary modules.
- Since MATLAB overwrites the path, set the Node path first: `setenv('PATH', [getenv('PATH') ':/path/to/node/bin']);`
  - e.g. `setenv('PATH', [getenv('PATH') ':/Users/rudotriton/.nvm/versions/node/v14.2.0/bin']);`
- `dir` calls in code may need to be replaced with `ls` for Windows. Also the code uses the `.name` field of the returned structs. `ls` may return a struct with different properties, in which case the `folder.name` may not be available and needs to be changed.
- `Evaluation.m` is the main file which runs and evaluates each method: template matching, eigenfaces and neural net.

---

- Training data is expected to be in: `./FaceDatabase/Train/<label>/img.jpg`, where `<label>` is a directory containing a training image which then gets labelled according to the directory name.
- Test data is expected to be in `./FaceDatabase/Train` as just `.jpg` images.

## Face Recognition

- `FaceRecognition` - A simple face recognition method using cross-correlation based template matching.
- `FaceRecognition1` - A face recognition using eigenfaces.
  - `EigenWDetect` - this is a spin-off based on the same method, however, it tries to first detect a face in the image based on the `viola-jones` method.
- `FaceRecognition2.js` - based on [face-api.js](https://github.com/justadudewhohacks/face-api.js), this method, detects faces, extracts landmarks, and builds a matcher based on them for relatively accurate face recognition.
