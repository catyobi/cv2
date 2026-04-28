let video = document.getElementById('videoInput');
let canvasOutput = document.getElementById('canvasOutput');
let virtualBackground = document.getElementById('virtualBackground');
let statusMessage = document.getElementById('statusMessage');

let streaming = false;
let src = null;
let dst = null;
let hsv = null;
let mask = null;
let low = null;
let high = null;
let cap = null;

// Default HSV Thresholds for a green screen
let hsvVals = {
    hMin: 35,
    hMax: 85,
    sMin: 40,
    sMax: 255,
    vMin: 40,
    vMax: 255
};

// Update UI Sliders
function updateSliders() {
    ['hMin', 'hMax', 'sMin', 'sMax', 'vMin', 'vMax'].forEach(id => {
        document.getElementById(id).value = hsvVals[id];
        document.getElementById(`${id}Val`).textContent = hsvVals[id];
    });
}

const controls = ['hMin', 'hMax', 'sMin', 'sMax', 'vMin', 'vMax'];
controls.forEach(id => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(`${id}Val`);
    el.addEventListener('input', (e) => {
        hsvVals[id] = parseInt(e.target.value);
        valEl.textContent = e.target.value;
    });
});

// Background selection
document.querySelectorAll('.bg-option').forEach(option => {
    option.addEventListener('click', (e) => {
        document.querySelectorAll('.bg-option').forEach(opt => opt.classList.remove('active'));
        const target = e.currentTarget;
        target.classList.add('active');
        const bg = target.getAttribute('data-bg');
        if (bg.startsWith('gradient')) {
            virtualBackground.style.backgroundImage = 'none';
            if (bg === 'gradient-1') {
                virtualBackground.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            } else {
                virtualBackground.style.background = 'linear-gradient(135deg, #f6d365 0%, #fda085 100%)';
            }
        } else {
            virtualBackground.style.background = '';
            virtualBackground.style.backgroundImage = `url('${bg}')`;
        }
    });
});

// Click on Canvas to pick color
canvasOutput.addEventListener('click', (e) => {
    if (!streaming || !hsv) return;
    const rect = canvasOutput.getBoundingClientRect();
    const scaleX = canvasOutput.width / rect.width;
    const scaleY = canvasOutput.height / rect.height;
    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);
    
    // Read the HSV value at that pixel
    const pixel = hsv.ucharPtr(y, x);
    const h = pixel[0];
    const s = pixel[1];
    const v = pixel[2];
    
    // Set a range around the picked color
    hsvVals.hMin = Math.max(0, h - 15);
    hsvVals.hMax = Math.min(179, h + 15);
    hsvVals.sMin = Math.max(0, s - 40);
    hsvVals.sMax = 255;
    hsvVals.vMin = Math.max(0, v - 40);
    hsvVals.vMax = 255;
    
    updateSliders();
});

function onOpenCvReady() {
    statusMessage.textContent = 'OpenCV.js is ready. Requesting camera...';
    statusMessage.className = 'loading';
    startCamera();
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function (err) {
            statusMessage.textContent = "Error accessing camera: " + err;
            statusMessage.className = '';
            console.error("Camera error:", err);
        });

    video.addEventListener('canplay', function (ev) {
        if (!streaming) {
            video.width = video.videoWidth;
            video.height = video.videoHeight;
            canvasOutput.width = video.videoWidth;
            canvasOutput.height = video.videoHeight;
            
            let maskPreview = document.getElementById('maskOutput');
            maskPreview.width = 320; // scaled down for debug
            maskPreview.height = 240;
            
            streaming = true;
            updateSliders();

            statusMessage.textContent = 'Camera active. Ready for Chroma Key.';
            statusMessage.className = 'active-status';
            initOpenCVVars();
            requestAnimationFrame(processVideo);
        }
    }, false);
}

function initOpenCVVars() {
    src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    hsv = new cv.Mat(video.height, video.width, cv.CV_8UC3);
    mask = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    low = new cv.Mat(video.height, video.width, hsv.type());
    high = new cv.Mat(video.height, video.width, hsv.type());
    cap = new cv.VideoCapture(video);
}

function processVideo() {
    try {
        if (!streaming) {
            cleanUp();
            return;
        }

        cap.read(src);
        src.copyTo(dst);

        // Convert to HSV for color masking
        cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

        // Create scalar bounds
        let lowScalar = new cv.Scalar(hsvVals.hMin, hsvVals.sMin, hsvVals.vMin);
        let highScalar = new cv.Scalar(hsvVals.hMax, hsvVals.sMax, hsvVals.vMax);

        low.setTo(lowScalar);
        high.setTo(highScalar);

        // Threshold the HSV image to get the background mask
        cv.inRange(hsv, low, high, mask);

        // Morphological operations to clean up the mask
        let M = cv.Mat.ones(5, 5, cv.CV_8U);
        cv.dilate(mask, mask, M, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
        cv.erode(mask, mask, M, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
        
        // Optional: blur the mask for softer edges
        cv.GaussianBlur(mask, mask, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

        // Debug mask output (scaled down)
        let maskDebug = new cv.Mat();
        cv.resize(mask, maskDebug, new cv.Size(320, 240), 0, 0, cv.INTER_AREA);
        cv.imshow('maskOutput', maskDebug);
        maskDebug.delete();

        // Make background transparent on the main canvas (dst)
        // mask is 255 for background, 0 for foreground
        // We want alpha = 0 for background, alpha = 255 for foreground
        // So alpha = 255 - mask
        let dstData = dst.data;
        let maskData = mask.data;
        for (let i = 0; i < maskData.length; i++) {
            // Set Alpha channel (every 4th byte: R, G, B, A)
            dstData[i * 4 + 3] = 255 - maskData[i]; 
        }

        // Draw transparent frame on canvas
        cv.imshow('canvasOutput', dst);
        
        M.delete();

        requestAnimationFrame(processVideo);
    } catch (err) {
        console.error("Error in processVideo:", err);
    }
}

function cleanUp() {
    if (src != null) src.delete();
    if (dst != null) dst.delete();
    if (hsv != null) hsv.delete();
    if (mask != null) mask.delete();
    if (low != null) low.delete();
    if (high != null) high.delete();
}
