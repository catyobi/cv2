const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let model;
let stream;
let running = false;

// LOAD MODEL
async function loadModel() {
    model = await blazeface.load();
    console.log("Model loaded");
}

// START CAMERA
async function startCamera() {
    try {
        if (!model) {
            await loadModel();
        }

        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            running = true;
            detectLoop(); // 🔥 new loop
        };

    } catch (error) {
        console.error(error);
    }
}

// STOP CAMERA
function stopCamera() {
    running = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
}

// 🔥 NEW DETECTION LOOP (stable)
function detectLoop() {
    if (!running) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (model) {
        model.estimateFaces(video, false).then(predictions => {

            ctx.strokeStyle = "lime";
            ctx.lineWidth = 3;

            predictions.forEach(prediction => {
                const [x, y] = prediction.topLeft;
                const [x2, y2] = prediction.bottomRight;

                ctx.strokeRect(x, y, x2 - x, y2 - y);
            });

        });
    }

    // 🔥 run every 100ms (stable + enough)
    setTimeout(detectLoop, 100);
}

// SCREENSHOT
function takeScreenshot() {
    if (!canvas.width) {
        alert("Camera not ready");
        return;
    }

    const link = document.createElement("a");
    link.download = "screenshot.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
}

// INIT
loadModel();