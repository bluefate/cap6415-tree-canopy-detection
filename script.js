// Form submission handler
document.getElementById('submissionForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    // Validate form
    if (!validateForm()) {
        return;
    }
    
    // Collect form data
    const formData = new FormData(this);
    const submissionData = {};
    
    // Convert FormData to object
    for (let [key, value] of formData.entries()) {
        if (key === 'submissionFile') {
            const file = document.getElementById('submissionFile').files[0];
            submissionData[key] = {
                name: file.name,
                size: file.size,
                type: file.type
            };
        } else {
            submissionData[key] = value;
        }
    }
    
    // Add timestamp
    submissionData.timestamp = new Date().toISOString();
    
    // Log submission data (in a real application, this would be sent to a server)
    console.log('Submission Data:', submissionData);
    
    // Store in localStorage for demo purposes
    const submissions = JSON.parse(localStorage.getItem('submissions') || '[]');
    submissions.push(submissionData);
    localStorage.setItem('submissions', JSON.stringify(submissions));
    
    // Show confirmation message
    showConfirmation();
});

// Form validation
function validateForm() {
    const form = document.getElementById('submissionForm');
    
    // Check required fields
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim() && field.type !== 'checkbox') {
            showError(field, 'This field is required');
            isValid = false;
        } else if (field.type === 'checkbox' && !field.checked) {
            showError(field, 'You must accept the terms and conditions');
            isValid = false;
        } else {
            clearError(field);
        }
    });
    
    // Validate email format
    const emailField = document.getElementById('email');
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (emailField.value && !emailPattern.test(emailField.value)) {
        showError(emailField, 'Please enter a valid email address');
        isValid = false;
    }
    
    // Validate file upload
    const fileInput = document.getElementById('submissionFile');
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const maxSize = 50 * 1024 * 1024; // 50MB
        
        if (file.size > maxSize) {
            showError(fileInput, 'File size must be less than 50MB');
            isValid = false;
        }
        
        if (!file.name.toLowerCase().endsWith('.csv')) {
            showError(fileInput, 'Only CSV files are allowed');
            isValid = false;
        }
    }
    
    return isValid;
}

// Show error message
function showError(field, message) {
    // Remove existing error message
    clearError(field);
    
    // Create error message element
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.color = '#dc3545';
    errorDiv.style.fontSize = '0.875em';
    errorDiv.style.marginTop = '5px';
    errorDiv.textContent = message;
    
    // Add error styling to field
    field.style.borderColor = '#dc3545';
    
    // Insert error message after field
    field.parentNode.appendChild(errorDiv);
}

// Clear error message
function clearError(field) {
    const errorDiv = field.parentNode.querySelector('.error-message');
    if (errorDiv) {
        errorDiv.remove();
    }
    field.style.borderColor = '';
}

// Show confirmation message
function showConfirmation() {
    const form = document.getElementById('submissionForm');
    const confirmation = document.getElementById('confirmationMessage');
    
    form.style.display = 'none';
    confirmation.style.display = 'block';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// File input change handler
document.getElementById('submissionFile').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const fileSize = (file.size / (1024 * 1024)).toFixed(2);
        console.log(`File selected: ${file.name} (${fileSize} MB)`);
    }
});

// Clear form on reset
document.getElementById('submissionForm').addEventListener('reset', function() {
    // Clear all error messages
    const errorMessages = document.querySelectorAll('.error-message');
    errorMessages.forEach(msg => msg.remove());
    
    // Reset field styling
    const inputs = document.querySelectorAll('input, textarea, select');
    inputs.forEach(input => {
        input.style.borderColor = '';
    });
});

// Add real-time validation
const formInputs = document.querySelectorAll('input[required], textarea[required], select[required]');
formInputs.forEach(input => {
    input.addEventListener('blur', function() {
        if (this.value.trim() === '' && this.type !== 'checkbox') {
            showError(this, 'This field is required');
        } else if (this.type === 'email') {
            const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailPattern.test(this.value)) {
                showError(this, 'Please enter a valid email address');
            } else {
                clearError(this);
            }
        } else {
            clearError(this);
        }
    });
    
    input.addEventListener('input', function() {
        if (this.value.trim() !== '') {
            clearError(this);
        }
    });
});

// Console message for developers
console.log('%cTree Canopy Detection Competition', 'font-size: 20px; font-weight: bold; color: #667eea;');
console.log('%cSubmission Form Ready', 'font-size: 14px; color: #28a745;');
console.log('View submissions in localStorage:', JSON.parse(localStorage.getItem('submissions') || '[]'));
