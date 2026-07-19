let videoStream = null;
let isStreaming = false;

let videoStream3 = null;
let isStreaming3 = false;

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
        if (e.target.files.length === 0) return;
        
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

    // 4. Fase 3 - Subida de Imagen
    const imageUpload3 = document.getElementById('imageUpload3');
    imageUpload3.addEventListener('change', (e) => {
        if (e.target.files.length === 0) return;
        
        stopCamera3();
        const file = e.target.files[0];
        const imgUrl = URL.createObjectURL(file);
        
        const img = new Image();
        img.onload = () => { processImage3(img); };
        img.src = imgUrl;
    });

    // 5. Fase 3 - Cámara
    const cameraBtn3 = document.getElementById('cameraBtn3');
    const video3 = document.getElementById('videoElement3');

    cameraBtn3.addEventListener('click', async () => {
        if (isStreaming3) {
            captureFrame3();
            stopCamera3();
            cameraBtn3.textContent = 'Usar Cámara';
            cameraBtn3.classList.remove('primary-btn');
            cameraBtn3.classList.add('secondary-btn');
        } else {
            try {
                videoStream3 = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video3.srcObject = videoStream3;
                video3.play();
                isStreaming3 = true;
                cameraBtn3.textContent = '📸 Tomar Foto';
                cameraBtn3.classList.remove('secondary-btn');
                cameraBtn3.classList.add('primary-btn');
                
                ['canvasHu', 'canvasHog', 'canvasBrisk'].forEach(id => {
                    document.getElementById(id).classList.remove('loaded');
                });
                
                requestAnimationFrame(() => processVideoLive3(video3));
            } catch (err) {
                alert('No se pudo acceder a la cámara: ' + err.message);
            }
        }
    });
});

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

async function runPipelineFromCanvas(sourceCanvas) {
    sourceCanvas.classList.add('loaded');
    
    try {
        const base64Img = sourceCanvas.toDataURL('image/jpeg');
        
        const response = await fetch('http://127.0.0.1:5000/preprocess_demo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Img })
        });
        
        if (!response.ok) {
            console.error("Error del servidor:", await response.text());
            return;
        }
        
        const data = await response.json();
        
        const drawImageToCanvas = (base64, canvasId) => {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = () => {
                canvas.width = sourceCanvas.width;
                canvas.height = sourceCanvas.height;
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                canvas.classList.add('loaded');
            };
            img.src = base64;
        };

        drawImageToCanvas(data.gray, 'canvasGray');
        drawImageToCanvas(data.clahe, 'canvasClahe');
        drawImageToCanvas(data.bilateral, 'canvasBilateral');
        drawImageToCanvas(data.threshold, 'canvasThreshold');
        
    } catch (err) {
        console.error("Error conectando con Backend: ", err);
    }
}

function stopCamera3() {
    if (videoStream3) {
        videoStream3.getTracks().forEach(track => track.stop());
        videoStream3 = null;
    }
    isStreaming3 = false;
}

function processVideoLive3(video) {
    if (!isStreaming3) return;
    
    const canvasBase3 = document.getElementById('canvasBase3');
    const ctx = canvasBase3.getContext('2d');
    canvasBase3.width = video.videoWidth;
    canvasBase3.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvasBase3.width, canvasBase3.height);
    
    canvasBase3.classList.add('loaded');
    
    requestAnimationFrame(() => processVideoLive3(video));
}

function captureFrame3() {
    runPipelineDescriptors(document.getElementById('canvasBase3'));
}

function processImage3(imgElement) {
    const canvasBase3 = document.getElementById('canvasBase3');
    const ctx = canvasBase3.getContext('2d', { willReadFrequently: true });
    canvasBase3.width = imgElement.width;
    canvasBase3.height = imgElement.height;
    ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);
    
    runPipelineDescriptors(canvasBase3);
}

async function runPipelineDescriptors(sourceCanvas) {
    sourceCanvas.classList.add('loaded');
    
    try {
        const base64Img = sourceCanvas.toDataURL('image/jpeg');
        
        const response = await fetch('http://127.0.0.1:5000/extract_demo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Img })
        });
        
        if (!response.ok) {
            console.error("Error del servidor:", await response.text());
            return;
        }
        
        const data = await response.json();
        
        const drawImageToCanvas = (base64, canvasId) => {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = () => {
                canvas.width = sourceCanvas.width;
                canvas.height = sourceCanvas.height;
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                canvas.classList.add('loaded');
            };
            img.src = base64;
        };

        drawImageToCanvas(data.base, 'canvasBase3');
        drawImageToCanvas(data.hu, 'canvasHu');
        drawImageToCanvas(data.hog, 'canvasHog');
        drawImageToCanvas(data.brisk, 'canvasBrisk');
        
    } catch (err) {
        console.error("Error conectando con Backend: ", err);
    }
}
