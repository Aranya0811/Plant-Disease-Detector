// Plant Disease Detector — Frontend Logic
// Step 1: Element selectors, File selection, and Preview UI Handlers

document.addEventListener('DOMContentLoaded', () => {
    // DOM Element Selectors
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

    // Trigger file input click when browse button is clicked
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Avoid triggering drop-zone click
        fileInput.click();
    });

    // Trigger file input click when drop zone is clicked (if no file is selected)
    dropZone.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    // Handle file selection from input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and Drop Event Listeners
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

    // File selection core handler
    function handleFileSelect(file) {
        // Validate it's an image
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (PNG or JPG).');
            return;
        }

        selectedFile = file;

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            diagnoseBtn.removeAttribute('disabled');
        };
        reader.readAsDataURL(file);
    }

    // Remove image handler
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Avoid triggering drop-zone click
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
});
