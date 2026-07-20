let mlVideoStream = null;
let mlIsStreaming = false;

document.addEventListener('DOMContentLoaded', () => {
    // 1. Manejo de submenú ML
    const mlNavItems = document.querySelectorAll('.sidebar li');
    const sections = document.querySelectorAll('.phase-section');

    mlNavItems.forEach(item => {
        if (!item.hasAttribute('data-target')) return; // ignorar enlaces externos

        item.addEventListener('click', () => {
            mlNavItems.forEach(nav => nav.classList.remove('active'));
            sections.forEach(sec => sec.style.display = 'none');
            
            item.classList.add('active');
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).style.display = 'block';
        });
    });

    // 2. Modal de Imágenes
    window.openModal = function(src, alt) {
        const modal = document.getElementById("imageModal");
        const modalImg = document.getElementById("expandedImg");
        const captionText = document.getElementById("modalCaption");
        modal.style.display = "block";
        modalImg.src = src;
        captionText.innerHTML = alt;
    };

    window.closeModal = function() {
        document.getElementById("imageModal").style.display = "none";
    };

    setupMLDemo();
});

// ==========================================
// DEMOSTRACIÓN MACHINE LEARNING (Flask API)
// ==========================================
function setupMLDemo() {
    const mlImageUpload = document.getElementById('ml-imageUpload');
    const mlCameraBtn = document.getElementById('ml-cameraBtn');
    const mlCaptureBtn = document.getElementById('ml-captureBtn');
    const mlVideo = document.getElementById('ml-videoElement');
    const mlImagePreview = document.getElementById('ml-imagePreview');
    const mlCanvasOriginal = document.getElementById('ml-canvasOriginal');

    mlImageUpload?.addEventListener('change', (e) => {
        if (e.target.files.length === 0) return;
        stopMLCamera();
        const file = e.target.files[0];
        
        const reader = new FileReader();
        reader.onload = function(event) {
            mlImagePreview.src = event.target.result;
            mlImagePreview.style.display = 'block';
            mlVideo.style.display = 'none';
            // Send to Flask
            predictWithFlask(event.target.result);
        };
        reader.readAsDataURL(file);
    });

    mlCameraBtn?.addEventListener('click', async () => {
        if (mlIsStreaming) {
            stopMLCamera();
            mlCameraBtn.textContent = 'Encender Cámara';
            mlCameraBtn.className = 'btn secondary-btn';
            mlCaptureBtn.style.display = 'none';
        } else {
            try {
                mlVideoStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                mlVideo.srcObject = mlVideoStream;
                mlVideo.play();
                mlIsStreaming = true;
                mlVideo.style.display = 'block';
                mlImagePreview.style.display = 'none';
                
                mlCameraBtn.textContent = 'Apagar Cámara';
                mlCameraBtn.className = 'btn secondary-btn';
                mlCaptureBtn.style.display = 'inline-block';
            } catch (err) {
                alert('No se pudo acceder a la cámara: ' + err.message);
            }
        }
    });

    mlCaptureBtn?.addEventListener('click', () => {
        if(!mlIsStreaming) return;
        mlCanvasOriginal.width = mlVideo.videoWidth;
        mlCanvasOriginal.height = mlVideo.videoHeight;
        const ctx = mlCanvasOriginal.getContext('2d');
        ctx.drawImage(mlVideo, 0, 0, mlCanvasOriginal.width, mlCanvasOriginal.height);
        
        const base64Img = mlCanvasOriginal.toDataURL('image/jpeg');
        predictWithFlask(base64Img);
    });

    const modelSelect = document.getElementById('modelSelect');
    const extractorSelect = document.getElementById('extractorSelect');
    const badge = document.getElementById('activeModelBadge');
    const box = document.getElementById('dynamic-result-box');
    const nameEl = document.getElementById('dynamic-model-name');
    const reevaluateBtn = document.getElementById('ml-reevaluateBtn');
    
    let lastImageBase64 = null;

    // Manejar cambio de modelo
    modelSelect?.addEventListener('change', (e) => {
        const isCNN = e.target.value === 'cnn';
        extractorSelect.disabled = isCNN;
        extractorSelect.style.opacity = isCNN ? '0.5' : '1';
        
        // Actualizar UI cosmética
        box.className = `prediction-box box-${e.target.value}`;
        const modelNames = {
            'cnn': 'CNN (Deep Learning)',
            'svm': 'SVM (Máquina de Soporte Vectorial)',
            'rf': 'Random Forest',
            'kmeans': 'K-Means (Clustering)'
        };
        nameEl.textContent = modelNames[e.target.value];
        
        let badgeText = `Modelo: ${e.target.options[e.target.selectedIndex].text}`;
        if(!isCNN) {
            badgeText += ` + ${extractorSelect.options[extractorSelect.selectedIndex].text}`;
        }
        badge.textContent = badgeText;
        
        // Limpiar resultado anterior para invitar a reevaluar
        document.getElementById('dynamic-class').textContent = 'Esperando ejecución...';
        document.getElementById('dynamic-conf').textContent = '--%';
        
        // Toggle K-Means Reference Table
        document.getElementById('kmeans-reference').style.display = (e.target.value === 'kmeans') ? 'block' : 'none';
    });

    // Manejar cambio de extractor
    extractorSelect?.addEventListener('change', (e) => {
        if(modelSelect.value !== 'cnn') {
            badge.textContent = `Modelo: ${modelSelect.options[modelSelect.selectedIndex].text} + ${e.target.options[e.target.selectedIndex].text}`;
            // Limpiar resultado anterior para invitar a reevaluar
            document.getElementById('dynamic-class').textContent = 'Esperando ejecución...';
            document.getElementById('dynamic-conf').textContent = '--%';
        }
    });

    // Sobrescribir captura/subida para guardar la última imagen
    const originalPredict = predictWithFlask;
    predictWithFlask = async function(base64Img) {
        lastImageBase64 = base64Img;
        reevaluateBtn.style.display = 'block';
        await originalPredict(base64Img);
    };

    reevaluateBtn?.addEventListener('click', () => {
        if(lastImageBase64) {
            predictWithFlask(lastImageBase64);
        }
    });

}

function stopMLCamera() {
    if (mlVideoStream) {
        mlVideoStream.getTracks().forEach(track => track.stop());
        mlVideoStream = null;
    }
    mlIsStreaming = false;
    document.getElementById('ml-videoElement').style.display = 'none';
}

async function predictWithFlask(base64Img) {
    const statusEl = document.getElementById('ml-status');
    const model = document.getElementById('modelSelect').value;
    const extractor = document.getElementById('extractorSelect').value;
    
    statusEl.innerHTML = 'Enviando a Flask... ⏳';
    
    // Resetear UI
    const classEl = document.getElementById('dynamic-class');
    const confEl = document.getElementById('dynamic-conf');
    const distContainer = document.getElementById('distribution-container');
    
    classEl.textContent = 'Analizando...';
    classEl.style.color = ''; // Reset color
    confEl.textContent = '--%';
    if (distContainer) distContainer.style.display = 'none';

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                image: base64Img,
                model: model,
                extractor: extractor
            })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Error en el servidor');
        }
        
        const data = await response.json();
        
        // Actualizar UI
        statusEl.innerHTML = 'Predicción Completada ✅';
        
        classEl.textContent = data.clase;
        if(model === 'kmeans') {
            confEl.textContent = 'No Supervisado';
            if(distContainer) distContainer.style.display = 'none';
        } else {
            confEl.textContent = `Accuracy: ${(data.prob * 100).toFixed(2)}%`;
            
            // Renderizar Distribución de Probabilidades
            if (data.distribucion && distContainer) {
                const distBars = document.getElementById('distribution-bars');
                distContainer.style.display = 'block';
                distBars.innerHTML = ''; 
                
                // Ordenar de mayor a menor probabilidad
                const sortedClasses = Object.keys(data.distribucion).sort((a, b) => data.distribucion[b] - data.distribucion[a]);
                
                sortedClasses.forEach(className => {
                    const prob = data.distribucion[className];
                    const percentage = (prob * 100).toFixed(1);
                    
                    const isWinner = className === data.clase;
                    const barColor = isWinner ? 'var(--primary)' : 'rgba(255, 255, 255, 0.4)';
                    const textColor = isWinner ? 'white' : '#aaa';
                    
                    const barHtml = `
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: ${textColor}; margin-bottom: 3px;">
                                <span>${className}</span>
                                <span style="font-weight: ${isWinner ? 'bold' : 'normal'};">${percentage}%</span>
                            </div>
                            <div style="width: 100%; background: rgba(255, 255, 255, 0.1); border-radius: 4px; height: 8px; overflow: hidden;">
                                <div style="width: ${percentage}%; background: ${barColor}; height: 100%; border-radius: 4px; transition: width 0.5s ease;"></div>
                            </div>
                        </div>
                    `;
                    distBars.innerHTML += barHtml;
                });
            }
        }

    } catch(err) {
        console.error(err);
        if (err.message === "No face detected") {
            statusEl.innerHTML = '❌ <span style="color: #ff4444;">No se detectó ningún rostro.</span>';
            classEl.textContent = 'Rostro no encontrado';
            classEl.style.color = '#ff4444';
            confEl.textContent = 'Por favor, apunta la cámara a tu cara.';
        } else {
            statusEl.innerHTML = '❌ Error de Conexión o Modelo no disponible.';
            classEl.textContent = 'Error';
            confEl.textContent = err.message;
        }
    }
}
