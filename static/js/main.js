document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.querySelector('.file-input-wrapper');
    const fileInput = document.querySelector('.file-input');
    const form = document.querySelector('.upload-form');
    const loading = document.querySelector('.loading');

    const progressFill = document.getElementById('progressFill'); // Ensure this element exists in upload.html

    // Drag and drop functionality
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        fileInput.files = e.dataTransfer.files;
        updateFileName();
    });

    // Click to upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', updateFileName);

    // Form submission with progress bar
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (file) {
            const formData = new FormData(form);
            loading.style.display = 'block';
            progressFill.style.display = 'block'; // Ensure progressFill is correctly selected

            fetch('/', {
                method: 'POST',
                body: formData
            }).then(response => response.text())
              .then(html => {
                  document.open();
                  document.write(html);
                  document.close();
              });

            // Simulate progress (replace with actual progress if possible)
            let width = 0;
            const interval = setInterval(() => {
                if (width >= 100) {
                    clearInterval(interval);
                } else {
                    width += 10;
                    progressFill.style.width = width + '%';
                }
            }, 300);
        }
    });

    function updateFileName() {
        const file = fileInput.files[0];
        if (file) {
            dropZone.querySelector('span').textContent = file.name;
        }
    }
});