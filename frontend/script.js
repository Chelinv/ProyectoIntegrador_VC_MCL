let cvReady = false;
let videoStream = null;
let isStreaming = false;

document.addEventListener('DOMContentLoaded', () => {
    // 1. Manejo del menú lateral
    const navItems = document.querySelectorAll('.sidebar li');
    const sections = document.querySelectorAll('.phase-section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(nav => nav.classList.remove('active'));
            sections.forEach(sec => sec.classList.remove('active'));

            item.classList.add('active');
            
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // 2. Manejo de Subida de Imagen
    const imageUpload = document.getElementById('imageUpload');
    imageUpload.addEventListener('change', (e) => {
        if (!cvReady || e.target.files.length === 0) return;
        
        stopCamera();
        const file = e.target.files[0];
        const imgUrl = URL.createObjectURL(file);
        
        const img = new Image();
        img.onload = () => { processImage(img); };
        img.src = imgUrl;
    });

    // 3. Manejo de Cámara
    const cameraBtn = document.getElementById('cameraBtn');
    const video = document.getElementById('videoElement');

    cameraBtn.addEventListener('click', async () => {
        if (!cvReady) return;

        if (isStreaming) {
            // Tomar foto
            captureFrame();
            stopCamera();
            cameraBtn.textContent = 'Usar Cámara';
            cameraBtn.classList.remove('primary-btn');
            cameraBtn.classList.add('secondary-btn');
        } else {
            // Iniciar cámara
            try {
                videoStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video.srcObject = videoStream;
                video.play();
                isStreaming = true;
                cameraBtn.textContent = '📸 Tomar Foto';
                cameraBtn.classList.remove('secondary-btn');
                cameraBtn.classList.add('primary-btn');
                
                // Limpiar canvas anteriores
                ['canvasClahe', 'canvasBilateral', 'canvasThreshold'].forEach(id => {
                    document.getElementById(id).classList.remove('loaded');
                });
                
                requestAnimationFrame(() => processVideoLive(video));
            } catch (err) {
                alert('No se pudo acceder a la cámara: ' + err.message);
            }
        }
    });
});

// Callback de OpenCV
function onOpenCvReady() {
    cv['onRuntimeInitialized'] = () => {
        cvReady = true;
        const statusEl = document.getElementById('status');
        statusEl.innerHTML = 'OpenCV.js Listo ✅';
        statusEl.classList.add('ready');
        document.getElementById('cameraBtn').disabled = false;
    };
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    isStreaming = false;
}

function processVideoLive(video) {
    if (!isStreaming) return;
    
    const canvasOriginal = document.getElementById('canvasOriginal');
    const ctx = canvasOriginal.getContext('2d');
    canvasOriginal.width = video.videoWidth;
    canvasOriginal.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvasOriginal.width, canvasOriginal.height);
    
    canvasOriginal.classList.add('loaded');
    
    requestAnimationFrame(() => processVideoLive(video));
}

function captureFrame() {
    runPipelineFromCanvas(document.getElementById('canvasOriginal'));
}

function processImage(imgElement) {
    const canvasOriginal = document.getElementById('canvasOriginal');
    const ctx = canvasOriginal.getContext('2d', { willReadFrequently: true });
    canvasOriginal.width = imgElement.width;
    canvasOriginal.height = imgElement.height;
    ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);
    
    runPipelineFromCanvas(canvasOriginal);
}

function runPipelineFromCanvas(sourceCanvas) {
    sourceCanvas.classList.add('loaded');
    
    try {
        let src = cv.imread(sourceCanvas);
        let gray = new cv.Mat();
        
        // A. Escala de grises
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
        const canvasGray = document.getElementById('canvasGray');
        cv.imshow(canvasGray, gray);
        canvasGray.classList.add('loaded');

        // B. CLAHE
        let claheMat = new cv.Mat();
        let clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
        clahe.apply(gray, claheMat);
        
        const canvasClahe = document.getElementById('canvasClahe');
        cv.imshow(canvasClahe, claheMat);
        canvasClahe.classList.add('loaded');

        // C. Filtro Bilateral
        let bilateralMat = new cv.Mat();
        cv.bilateralFilter(claheMat, bilateralMat, 9, 75, 75, cv.BORDER_DEFAULT);
        
        const canvasBilateral = document.getElementById('canvasBilateral');
        cv.imshow(canvasBilateral, bilateralMat);
        canvasBilateral.classList.add('loaded');

        // D. Umbralización Adaptativa
        let threshMat = new cv.Mat();
        cv.adaptiveThreshold(bilateralMat, threshMat, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);
        
        const canvasThreshold = document.getElementById('canvasThreshold');
        cv.imshow(canvasThreshold, threshMat);
        canvasThreshold.classList.add('loaded');

        // Limpiar
        src.delete(); gray.delete(); claheMat.delete(); 
        clahe.delete(); bilateralMat.delete(); threshMat.delete();

    } catch (err) {
        console.error("Error OpenCV: ", err);
    }
}
