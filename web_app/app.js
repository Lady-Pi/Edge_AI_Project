let ageModel, genderModel, emotionModel;

async function loadModels() {
  ageModel = await tf.loadLayersModel('models/age/model.json');
  genderModel = await tf.loadLayersModel('models/gender/model.json');
  emotionModel = await tf.loadLayersModel('models/emotion/model.json');
  console.log("✅ Models loaded");
}

async function setupWebcam() {
  const webcam = document.getElementById('webcam');
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcam.srcObject = stream;
}

function preprocessImage(canvas, size, isGray = false) {
  let img = tf.browser.fromPixels(canvas);
  if (isGray) {
    img = tf.image.rgbToGrayscale(img);
  }
  return tf.tidy(() =>
    img.resizeNearestNeighbor([size, size])
       .expandDims(0)
       .div(255.0)
  );
}

async function predict() {
  const webcam = document.getElementById('webcam');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

  const ageInput = preprocessImage(canvas, 224);
  const genderInput = preprocessImage(canvas, 224);
  const emotionInput = preprocessImage(canvas, 48, true);

  const ageProb = (await ageModel.predict(ageInput).data())[0];
  const genderProb = (await genderModel.predict(genderInput).data())[0];
  const emotionProb = (await emotionModel.predict(emotionInput).data())[0];

  document.getElementById('age-result').innerText = ageProb > 0.5 ? 'Elderly' : 'Adult';
  document.getElementById('gender-result').innerText = genderProb > 0.5 ? 'Male' : 'Female';
  document.getElementById('emotion-result').innerText = emotionProb > 0.5 ? 'Sad' : 'Happy';

  ageInput.dispose();
  genderInput.dispose();
  emotionInput.dispose();
}

document.getElementById('predict-btn').addEventListener('click', predict);

window.addEventListener('load', async () => {
  await loadModels();
  await setupWebcam();
});
