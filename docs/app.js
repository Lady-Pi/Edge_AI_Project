let ageModel, genderModel, emotionModel;
let modelsLoaded = false;

// Check if browser supports required APIs
function checkBrowserSupport() {
  if (!('mediaDevices' in navigator) || !('getUserMedia' in navigator.mediaDevices)) {
    console.error('getUserMedia is not supported in this browser');
    alert('Your browser does not support camera access. Please use a modern browser like Chrome, Firefox, or Safari.');
    return false;
  }

  if (!window.tf) {
    console.error('TensorFlow.js is not loaded');
    alert('TensorFlow.js failed to load. Please check your internet connection.');
    return false;
  }

  return true;
}

async function loadModels() {
  try {
    console.log("🔄 Loading models...");

    // Load models with proper error handling
    ageModel = await tf.loadLayersModel('./models/age/model.json');
    console.log("✅ Age model loaded");

    genderModel = await tf.loadLayersModel('./models/gender/model.json');
    console.log("✅ Gender model loaded");

    emotionModel = await tf.loadLayersModel('./models/emotion/model.json');
    console.log("✅ Emotion model loaded");

    modelsLoaded = true;
    console.log("✅ All models loaded successfully");

    // Enable predict button
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.style.opacity = '1';
    }

  } catch (error) {
    console.error("❌ Error loading models:", error);
    alert('Failed to load AI models. Please check your internet connection and refresh the page.');
    modelsLoaded = false;
  }
}

async function setupWebcam() {
  console.log("🔄 Requesting camera access...");
  const webcam = document.getElementById('webcam');

  try {
    // Check for HTTPS
    if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
      throw new Error('Camera access requires HTTPS. Please use https:// instead of http://');
    }

    const constraints = {
      video: {
        width: { ideal: 320 },
        height: { ideal: 240 },
        facingMode: 'user' // Front-facing camera
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    console.log("✅ Camera stream obtained");

    webcam.srcObject = stream;

    // Wait for video to be ready
    webcam.addEventListener('loadedmetadata', () => {
      console.log("✅ Video metadata loaded");
    });

    // Show success message
    const statusDiv = document.getElementById('camera-status');
    if (statusDiv) {
      statusDiv.textContent = '✅ Camera ready';
      statusDiv.style.color = 'green';
    }

  } catch (error) {
    console.error("❌ Camera error:", error);

    let errorMessage = 'Camera access failed: ';

    if (error.name === 'NotAllowedError') {
      errorMessage += 'Permission denied. Please allow camera access and refresh the page.';
    } else if (error.name === 'NotFoundError') {
      errorMessage += 'No camera found. Please connect a camera and refresh the page.';
    } else if (error.name === 'NotSupportedError') {
      errorMessage += 'Camera not supported by this browser.';
    } else if (error.name === 'SecurityError') {
      errorMessage += 'Security error. Make sure you are using HTTPS.';
    } else {
      errorMessage += error.message;
    }

    alert(errorMessage);

    // Show error status
    const statusDiv = document.getElementById('camera-status');
    if (statusDiv) {
      statusDiv.textContent = '❌ Camera failed';
      statusDiv.style.color = 'red';
    }
  }
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
  if (!modelsLoaded) {
    alert('Models are still loading. Please wait...');
    return;
  }

  const webcam = document.getElementById('webcam');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  try {
    // Check if webcam is ready
    if (webcam.readyState !== 4) {
      alert('Camera is not ready. Please wait for the camera to load.');
      return;
    }

    console.log("🔄 Running prediction...");

    // Draw current frame to canvas
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

    // Preprocess images for each model
    const ageInput = preprocessImage(canvas, 224);
    const genderInput = preprocessImage(canvas, 224);
    const emotionInput = preprocessImage(canvas, 48, true);

    // Run predictions
    const agePred = await ageModel.predict(ageInput);
    const genderPred = await genderModel.predict(genderInput);
    const emotionPred = await emotionModel.predict(emotionInput);

    // Get prediction values
    const ageProb = (await agePred.data())[0];
    const genderProb = (await genderPred.data())[0];
    const emotionProb = (await emotionPred.data())[0];

    // Update results
    document.getElementById('age-result').innerText = ageProb > 0.5 ? 'Elderly' : 'Adult';
    document.getElementById('gender-result').innerText = genderProb > 0.5 ? 'Male' : 'Female';
    document.getElementById('emotion-result').innerText = emotionProb > 0.5 ? 'Sad' : 'Happy';

    console.log("✅ Prediction completed");
    console.log(`Age: ${ageProb > 0.5 ? 'Elderly' : 'Adult'} (${ageProb.toFixed(3)})`);
    console.log(`Gender: ${genderProb > 0.5 ? 'Male' : 'Female'} (${genderProb.toFixed(3)})`);
    console.log(`Emotion: ${emotionProb > 0.5 ? 'Sad' : 'Happy'} (${emotionProb.toFixed(3)})`);

    // Clean up tensors
    ageInput.dispose();
    genderInput.dispose();
    emotionInput.dispose();
    agePred.dispose();
    genderPred.dispose();
    emotionPred.dispose();

  } catch (error) {
    console.error("❌ Prediction error:", error);
    alert('Prediction failed: ' + error.message);
  }
}

// Initialize everything when page loads
window.addEventListener('load', async () => {
  console.log("🚀 Starting Edge AI Face Recognition");

  // Check browser support first
  if (!checkBrowserSupport()) {
    return;
  }

  // Disable predict button initially
  const predictBtn = document.getElementById('predict-btn');
  if (predictBtn) {
    predictBtn.disabled = true;
    predictBtn.style.opacity = '0.5';
  }

  // Load models and setup webcam simultaneously
  try {
    await Promise.all([
      loadModels(),
      setupWebcam()
    ]);
    console.log("✅ Initialization complete");
  } catch (error) {
    console.error("❌ Initialization failed:", error);
  }
});

// Add predict button event listener
document.addEventListener('DOMContentLoaded', () => {
  const predictBtn = document.getElementById('predict-btn');
  if (predictBtn) {
    predictBtn.addEventListener('click', predict);
  }
});