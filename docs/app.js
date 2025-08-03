let ageModel, genderModel, emotionModel;

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

async function testSingleModel(path, modelName) {
  try {
    console.log(`🔄 Testing ${modelName} at: ${path}`);

    // First check if the file exists
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`File not found: ${path} (Status: ${response.status})`);
    }

    console.log(`📁 ${modelName} file exists`);

    // Try to load the model
    const model = await tf.loadLayersModel(path);
    console.log(`✅ ${modelName} model loaded successfully`);

    return model;

  } catch (error) {
    console.error(`❌ Failed to load ${modelName}:`, error.message);
    throw error;
  }
}

async function loadModels() {
  try {
    console.log("🔄 Loading models...");
    console.log("Current location:", window.location.href);

    // Your models should be at these paths based on your GitHub structure
    const modelPaths = {
      age: './models/age/model.json',
      gender: './models/gender/model.json',
      emotion: './models/emotion/model.json'
    };

    // Load each model individually with detailed error reporting
    ageModel = await testSingleModel(modelPaths.age, 'Age');
    genderModel = await testSingleModel(modelPaths.gender, 'Gender');
    emotionModel = await testSingleModel(modelPaths.emotion, 'Emotion');

    // Mark models as loaded globally
    window.modelsLoaded = true;
    console.log("✅ All models loaded successfully");

    // Enable predict button
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.style.opacity = '1';
      predictBtn.textContent = '🔮 Predict';
    }

    // Update status
    const statusDiv = document.getElementById('model-status');
    if (statusDiv) {
      statusDiv.textContent = '✅ Models ready';
      statusDiv.style.color = 'green';
    }

  } catch (error) {
    console.error("❌ Error loading models:", error);

    // Show detailed error to user
    const errorMsg = `Failed to load AI models: ${error.message}\n\nPlease check:\n1. Model files are uploaded correctly\n2. File paths are correct\n3. Internet connection is stable`;
    alert(errorMsg);

    window.modelsLoaded = false;

    // Update status
    const statusDiv = document.getElementById('model-status');
    if (statusDiv) {
      statusDiv.textContent = '❌ Models failed to load';
      statusDiv.style.color = 'red';
    }
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
  if (!window.modelsLoaded) {
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

    // Update results with animation
    const ageResult = document.getElementById('age-result');
    const genderResult = document.getElementById('gender-result');
    const emotionResult = document.getElementById('emotion-result');

    ageResult.innerText = ageProb > 0.5 ? 'Elderly' : 'Adult';
    genderResult.innerText = genderProb > 0.5 ? 'Male' : 'Female';
    emotionResult.innerText = emotionProb > 0.5 ? 'Sad' : 'Happy';

    // Add animation class
    ageResult.classList.add('updated');
    genderResult.classList.add('updated');
    emotionResult.classList.add('updated');

    // Remove animation class after animation completes
    setTimeout(() => {
      ageResult.classList.remove('updated');
      genderResult.classList.remove('updated');
      emotionResult.classList.remove('updated');
    }, 600);

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
    predictBtn.textContent = '⏳ Loading...';
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