let ageModel, genderModel, emotionModel;
let currentStream = null;
let currentFacingMode = 'user';
let performanceLog = [];

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
    const startTime = performance.now();
    console.log(`🔄 Testing ${modelName} at: ${path}`);

    // First check if the file exists
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`File not found: ${path} (Status: ${response.status})`);
    }

    console.log(`📁 ${modelName} file exists`);

    // Try to load the model
    const model = await tf.loadLayersModel(path);
    const loadTime = performance.now() - startTime;

    console.log(`✅ ${modelName} model loaded successfully in ${loadTime.toFixed(2)}ms`);

    // Store performance data
    const modelSize = await fetch(path).then(r => r.headers.get('content-length') || 'Unknown');
    window.performanceData = window.performanceData || {};
    window.performanceData[modelName] = {
      loadTime: loadTime,
      size: modelSize,
      path: path
    };

    return model;

  } catch (error) {
    console.error(`❌ Failed to load ${modelName}:`, error.message);
    throw error;
  }
}

async function loadModels() {
  try {
    const totalStartTime = performance.now();
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

    const totalLoadTime = performance.now() - totalStartTime;
    console.log(`✅ All models loaded in ${totalLoadTime.toFixed(2)}ms`);

    // Mark models as loaded globally
    window.modelsLoaded = true;
    window.totalModelLoadTime = totalLoadTime;

    // Enable predict button
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.style.opacity = '1';
      predictBtn.textContent = 'Predict';
    }

    // Update status
    const statusDiv = document.getElementById('model-status');
    if (statusDiv) {
      statusDiv.textContent = `✅ Models ready (${totalLoadTime.toFixed(0)}ms)`;
      statusDiv.style.color = 'green';
    }

    // Show performance button
    const perfBtn = document.getElementById('performance-btn');
    if (perfBtn) {
      perfBtn.disabled = false;
      perfBtn.style.opacity = '1';
    }

  } catch (error) {
    console.error("❌ Error loading models:", error);

    const errorMsg = `Failed to load AI models: ${error.message}\n\nPlease check:\n1. Model files are uploaded correctly\n2. File paths are correct\n3. Internet connection is stable`;
    alert(errorMsg);

    window.modelsLoaded = false;

    const statusDiv = document.getElementById('model-status');
    if (statusDiv) {
      statusDiv.textContent = '❌ Models failed to load';
      statusDiv.style.color = 'red';
    }
  }
}

async function switchCamera() {
  currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';

  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }

  const switchBtn = document.getElementById('switch-camera-btn');
  if (switchBtn) {
    switchBtn.textContent = currentFacingMode === 'user' ? 'Switch to Back Camera' : 'Switch to Front Camera';
  }

  await setupWebcam();
}

async function setupWebcam() {
  console.log("🔄 Requesting camera access...");
  const webcam = document.getElementById('webcam');

  try {
    if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
      throw new Error('Camera access requires HTTPS. Please use https:// instead of http://');
    }

    const constraints = {
      video: {
        width: { ideal: 320 },
        height: { ideal: 240 },
        facingMode: currentFacingMode
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentStream = stream;
    console.log("✅ Camera stream obtained");

    webcam.srcObject = stream;

    webcam.addEventListener('loadedmetadata', () => {
      console.log("✅ Video metadata loaded");
    });

    const statusDiv = document.getElementById('camera-status');
    if (statusDiv) {
      statusDiv.textContent = `✅ Camera ready (${currentFacingMode === 'user' ? 'Front' : 'Back'})`;
      statusDiv.style.color = 'green';
    }

    const switchBtn = document.getElementById('switch-camera-btn');
    if (switchBtn) {
      switchBtn.disabled = false;
      switchBtn.style.opacity = '1';
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
    if (webcam.readyState !== 4) {
      alert('Camera is not ready. Please wait for the camera to load.');
      return;
    }

    const predictionStartTime = performance.now();
    console.log("🔄 Running prediction...");

    // Draw current frame to canvas
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

    // Preprocessing time
    const preprocessStartTime = performance.now();
    const ageInput = preprocessImage(canvas, 224);
    const genderInput = preprocessImage(canvas, 224);
    const emotionInput = preprocessImage(canvas, 48, true);
    const preprocessTime = performance.now() - preprocessStartTime;

    // Inference time
    const inferenceStartTime = performance.now();
    const agePred = await ageModel.predict(ageInput);
    const genderPred = await genderModel.predict(genderInput);
    const emotionPred = await emotionModel.predict(emotionInput);
    const inferenceTime = performance.now() - inferenceStartTime;

    // Get prediction values
    const ageProb = (await agePred.data())[0];
    const genderProb = (await genderPred.data())[0];
    const emotionProb = (await emotionPred.data())[0];

    const totalPredictionTime = performance.now() - predictionStartTime;

    // Store performance data
    const performanceData = {
      timestamp: new Date().toISOString(),
      totalTime: totalPredictionTime,
      preprocessTime: preprocessTime,
      inferenceTime: inferenceTime,
      predictions: {
        age: { value: ageProb, label: ageProb > 0.5 ? 'Elderly' : 'Adult' },
        gender: { value: genderProb, label: genderProb > 0.5 ? 'Male' : 'Female' },
        emotion: { value: emotionProb, label: emotionProb > 0.5 ? 'Sad' : 'Happy' }
      }
    };

    performanceLog.push(performanceData);
    window.performanceLog = performanceLog; // Make available globally

    // Update results with animation
    const ageResult = document.getElementById('age-result');
    const genderResult = document.getElementById('gender-result');
    const emotionResult = document.getElementById('emotion-result');

    ageResult.innerText = performanceData.predictions.age.label;
    genderResult.innerText = performanceData.predictions.gender.label;
    emotionResult.innerText = performanceData.predictions.emotion.label;

    // Add animation class
    ageResult.classList.add('updated');
    genderResult.classList.add('updated');
    emotionResult.classList.add('updated');

    setTimeout(() => {
      ageResult.classList.remove('updated');
      genderResult.classList.remove('updated');
      emotionResult.classList.remove('updated');
    }, 600);

    console.log("✅ Prediction completed");
    console.log(`⏱️ Total time: ${totalPredictionTime.toFixed(2)}ms`);
    console.log(`📊 Preprocessing: ${preprocessTime.toFixed(2)}ms, Inference: ${inferenceTime.toFixed(2)}ms`);
    console.log(`Age: ${performanceData.predictions.age.label} (${ageProb.toFixed(3)})`);
    console.log(`Gender: ${performanceData.predictions.gender.label} (${genderProb.toFixed(3)})`);
    console.log(`Emotion: ${performanceData.predictions.emotion.label} (${emotionProb.toFixed(3)})`);

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

function downloadCanvasImage() {
  const canvas = document.getElementById('canvas');
  const link = document.createElement('a');
  link.download = `webcam_capture_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}


function downloadPerformanceData() {
  if (!window.performanceLog || window.performanceLog.length === 0) {
    alert('No performance data available. Run some predictions first!');
    return;
  }

  // Create comprehensive performance report
  const report = {
    deviceInfo: {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      hardwareConcurrency: navigator.hardwareConcurrency,
      memory: navigator.deviceMemory || 'Unknown'
    },
    modelInfo: window.performanceData || {},
    totalModelLoadTime: window.totalModelLoadTime || 'Unknown',
    predictions: window.performanceLog,
    summary: {
      totalPredictions: window.performanceLog.length,
      averageInferenceTime: window.performanceLog.reduce((sum, log) => sum + log.inferenceTime, 0) / window.performanceLog.length,
      averageTotalTime: window.performanceLog.reduce((sum, log) => sum + log.totalTime, 0) / window.performanceLog.length
    }
  };

  // Download as JSON
  const dataStr = JSON.stringify(report, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);

  const link = document.createElement('a');
  link.href = url;
  link.download = `edge_ai_performance_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
  link.click();

  URL.revokeObjectURL(url);
  console.log('📥 Performance data downloaded');
}

// Initialize everything when page loads
window.addEventListener('load', async () => {
  console.log("🚀 Starting Edge AI Face Recognition");

  if (!checkBrowserSupport()) {
    return;
  }

  const predictBtn = document.getElementById('predict-btn');
  const switchBtn = document.getElementById('switch-camera-btn');
  const perfBtn = document.getElementById('performance-btn');

  if (predictBtn) {
    predictBtn.disabled = true;
    predictBtn.style.opacity = '0.5';
    predictBtn.textContent = 'Loading...';
  }

  if (switchBtn) {
    switchBtn.disabled = true;
    switchBtn.style.opacity = '0.5';
  }

  if (perfBtn) {
    perfBtn.disabled = true;
    perfBtn.style.opacity = '0.5';
  }

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

// Add event listeners
document.addEventListener('DOMContentLoaded', () => {
  const predictBtn = document.getElementById('predict-btn');
  const switchBtn = document.getElementById('switch-camera-btn');
  const perfBtn = document.getElementById('performance-btn');

  if (predictBtn) {
    predictBtn.addEventListener('click', predict);
  }

  if (switchBtn) {
    switchBtn.addEventListener('click', switchCamera);
  }

  if (perfBtn) {
    perfBtn.addEventListener('click', downloadPerformanceData);
  }
});