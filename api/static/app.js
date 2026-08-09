
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const diagnoseBtn = document.getElementById('diagnose-btn');
    const diagnoseSpinner = document.getElementById('diagnose-spinner');
    const resultsSection = document.getElementById('results-section');
    const predictedPlant = document.getElementById('predicted-plant');
    const predictedCondition = document.getElementById('predicted-condition');
    const confidenceCircle = document.getElementById('confidence-circle');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const confidenceVerdict = document.getElementById('confidence-verdict');
    const lowConfidenceWarning = document.getElementById('low-confidence-warning');
    const matchesList = document.getElementById('matches-list');
    const annotatedImage = document.getElementById('annotated-image');
    let selectedFile = null;
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    dropZone.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (PNG or JPG).');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            diagnoseBtn.removeAttribute('disabled');
        };
        reader.readAsDataURL(file);
    }
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage();
    });
    function clearImage() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.style.display = 'none';
        diagnoseBtn.setAttribute('disabled', 'true');
        resultsSection.style.display = 'none';
    }
    diagnoseBtn.addEventListener('click', () => {
        if (!selectedFile) return;
        diagnoseBtn.setAttribute('disabled', 'true');
        diagnoseSpinner.style.display = 'block';
        diagnoseBtn.querySelector('span').textContent = 'Analyzing Leaf...';
        resultsSection.style.display = 'none';
        const formData = new FormData();
        formData.append('file', selectedFile);
        fetch('/api/v1/predict/annotated', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('API server returned an error response.');
            }
            return response.json();
        })
        .then(data => {
            renderResults(data);
        })
        .catch(err => {
            console.error(err);
            alert('Failed to analyze the leaf. Please check that the backend server is running and try again.');
        })
        .finally(() => {
            diagnoseBtn.removeAttribute('disabled');
            diagnoseSpinner.style.display = 'none';
            diagnoseBtn.querySelector('span').textContent = 'Analyze Leaf';
        });
    });
    function renderResults(data) {
        predictedPlant.textContent = data.plant;
        predictedCondition.textContent = data.condition;
        const confidenceVal = Math.round(data.confidence * 100);
        confidencePercentage.textContent = `${confidenceVal}%`;
        confidenceCircle.setAttribute('stroke-dasharray', `${confidenceVal}, 100`);
        if (data.confidence >= 0.85) {
            confidenceVerdict.textContent = 'Excellent Match';
            confidenceVerdict.style.color = 'var(--color-primary)';
            confidenceCircle.style.stroke = 'var(--color-primary)';
        } else if (data.confidence >= 0.60) {
            confidenceVerdict.textContent = 'Good Match';
            confidenceVerdict.style.color = 'var(--color-warning)';
            confidenceCircle.style.stroke = 'var(--color-warning)';
        } else {
            confidenceVerdict.textContent = 'Weak Match';
            confidenceVerdict.style.color = 'var(--color-error)';
            confidenceCircle.style.stroke = 'var(--color-error)';
        }
        if (!data.is_confident) {
            lowConfidenceWarning.style.display = 'flex';
        } else {
            lowConfidenceWarning.style.display = 'none';
        }
        matchesList.innerHTML = '';
        data.top3.forEach(match => {
            const percentage = Math.round(match.confidence * 100);
            const matchItem = document.createElement('div');
            matchItem.className = 'match-item';
            matchItem.innerHTML = `
                <div class="match-info">
                    <span class="match-name">${match.display_name}</span>
                    <span class="match-value">${percentage}%</span>
                </div>
                <div class="match-bar-bg">
                    <div class="match-bar-fill" style="width: 0%;"></div>
                </div>
            `;
            matchesList.appendChild(matchItem);
            setTimeout(() => {
                const fillBar = matchItem.querySelector('.match-bar-fill');
                if (fillBar) {
                    fillBar.style.width = `${percentage}%`;
                    if (match.confidence >= 0.85) {
                        fillBar.style.background = 'var(--color-primary)';
                    } else if (match.confidence >= 0.60) {
                        fillBar.style.background = 'var(--color-warning)';
                    } else {
                        fillBar.style.background = 'var(--color-error)';
                    }
                }
            }, 50);
        });
        if (data.annotated_image_base64) {
            annotatedImage.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
        }
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});