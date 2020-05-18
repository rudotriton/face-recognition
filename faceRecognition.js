// require('@tensorflow/tfjs-node');

const faceapi = require('face-api.js');
const fs = require('fs');
const path = require('path');
const util = require('util');
const canvas = require('canvas');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const options = new faceapi.SsdMobilenetv1Options({
  minConfidence: 0.01,
  maxResults: 1,
});

// asynchronous readdir
const asyncReaddir = util.promisify(fs.readdir);
// asynchronous writeFile
const asyncWriteFile = util.promisify(fs.writeFile);

/**
 *
 * @param  {string} dirpath - path to the training images directory
 * @returns {Array} objects containing relative paths to images and labels
 *  [
 *    {
 *      fullpath: './FaceDatabase/Train/000001/000001.jpg',
 *      label: '000001'
 *    },
 *   ...
 *  ]
 */
const getAllTrainingFiles = async (dirpath) => {
  let files = await asyncReaddir(dirpath, {
    withFileTypes: true,
  });
  files = files.filter((file) => file.name !== '.DS_Store');
  let response = await Promise.all(
    files.map(async (dir) => {
      const file = fs.readdirSync(`${dirpath}/${dir.name}`)[0];
      return {
        fullpath: `${dirpath}/${dir.name}/${file}`,
        label: dir.name,
      };
    })
  );
  return response;
};

/**
 * loads pre-trained models, detect faces and landmarks on training images, and
 * extracts face descriptors
 *
 * @param  {string} dirpath - path to training data
 */
const loadTrainingData = async (dirpath) => {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models');
  await faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
  await faceapi.nets.faceRecognitionNet.loadFromDisk('./models');

  let trainingFiles = await getAllTrainingFiles(dirpath);
  return Promise.all(
    trainingFiles.map(async (file) => {
      const descriptors = [];
      const image = await canvas.loadImage(file.fullpath);
      const results = await faceapi
        .detectSingleFace(image, options)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (results === undefined) {
        return;
      }
      descriptors.push(results.descriptor);

      return new faceapi.LabeledFaceDescriptors(file.label, descriptors);
    })
  );
};

/**
 * Face recognition
 *
 * @param  {string} trainpath - path to directory containing directories of
 *   training images
 * @param  {string} testpath - path to directory containing test images
 */
const faceRecognition = async (trainpath, testpath) => {
  console.time('timer');
  let trainingData = await loadTrainingData(trainpath);
  // filter out images without detected faces
  trainingData = trainingData.filter((image) => image !== undefined);
  // construct a matcher against  which test images are compared
  const faceMatcher = new faceapi.FaceMatcher(trainingData);
  console.timeLog('timer');

  // read file names from the test files directory
  let testFileNames = await asyncReaddir(testpath);
  testFileNames = testFileNames.filter((file) => file !== '.DS_Store');
  // match each image against the matcher collection
  let results = await Promise.all(
    testFileNames.map(async (fileName, i) => {
      const testImage = await canvas.loadImage(`${testpath}/${fileName}`);
      const singleResult = await faceapi
        .detectSingleFace(testImage, options)
        .withFaceLandmarks()
        .withFaceDescriptor();

      // in case we fail to find a a face in the test image
      let bestMatch = '000000';
      // if test image has a detected face w/ landmarks and descriptors
      if (singleResult) {
        bestMatch = faceMatcher.findBestMatch(singleResult.descriptor);
        bestMatch._label =
          bestMatch._label === 'unknown' ? '000000' : bestMatch._label;
        bestMatch = bestMatch.toString();
        bestMatch = bestMatch.slice(0, 6);
      }
      return bestMatch;
    })
  );

  // write labels to a file
  await asyncWriteFile('results.csv', ['label', ...results].join('\n'));
  console.timeEnd('timer');
};

// construct absolute paths
const trainfilesPath = path.join(process.env.PWD, process.argv[2]);
const testfilesPath = path.join(process.env.PWD, process.argv[3]);

faceRecognition(trainfilesPath, testfilesPath);
