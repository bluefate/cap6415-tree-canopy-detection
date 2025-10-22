# Tree Canopy Detection Competition - Submission Guide

## Overview

This is the official submission form for the Tree Canopy Detection Competition. Use this form to submit your predictions and model information.

## How to Use

### Running the Form Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/bluefate/Tree-Canopy-Detection.git
   cd Tree-Canopy-Detection
   ```

2. Start a local web server:
   ```bash
   # Using Python 3
   python3 -m http.server 8080
   
   # Or using Python 2
   python -m SimpleHTTPServer 8080
   
   # Or using Node.js
   npx http-server -p 8080
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

### Filling Out the Form

The submission form consists of several sections:

#### 1. Participant Information (Required)
- **Full Name**: Your full name
- **Email Address**: Valid email address for confirmation
- **Affiliation/Organization**: Optional - your university, company, etc.

#### 2. Team Information (Optional)
- **Team Name**: If participating as a team, provide your team name
- **Team Members**: List all team member names (one per line)

#### 3. Submission Details (Required)
- **Prediction File (CSV)**: Upload your predictions in CSV format
  - Maximum file size: 50MB
  - Format: CSV only
  - Expected columns: `image_id`, `canopy_coverage` (or as specified in competition rules)
  
- **Model Name**: Give your model a descriptive name (e.g., "UNet-ResNet50")

- **Model Description**: Describe your approach, including:
  - Architecture used
  - Key features or techniques
  - Data preprocessing steps
  - Post-processing methods

- **Framework Used**: Select the ML framework (PyTorch, TensorFlow, Keras, etc.)

- **Approximate Training Time**: Help us understand computational requirements

#### 4. Additional Information
- **Additional Comments**: Any extra information about your submission
- **Terms and Conditions**: You must accept to submit (Required)
- **Code Sharing**: Optional - agree to share code after competition ends

## Submission File Format

Your prediction CSV file should follow this structure:

```csv
image_id,canopy_coverage
001,0.75
002,0.42
003,0.88
004,0.31
005,0.67
```

Where:
- `image_id`: Unique identifier for each image
- `canopy_coverage`: Predicted canopy coverage value (0.0 to 1.0)

## Validation Rules

The form validates the following:
- All required fields must be filled
- Email must be in valid format
- File must be CSV format
- File size must be under 50MB
- Terms and conditions must be accepted

## Data Storage

For demonstration purposes, submissions are stored in the browser's localStorage. In a production environment, this would be sent to a backend server for processing and evaluation.

## Technical Details

### Files
- `index.html`: Main form structure
- `styles.css`: Form styling and responsive design
- `script.js`: Form validation and submission logic

### Features
- Real-time form validation
- File upload with type and size checking
- Responsive design for mobile and desktop
- Confirmation page after successful submission
- Form reset capability

## Troubleshooting

### Form Won't Submit
- Ensure all required fields (marked with *) are filled
- Check that email address is valid
- Verify file is CSV format and under 50MB
- Make sure terms and conditions checkbox is checked

### File Upload Issues
- Only CSV files are accepted
- Maximum file size is 50MB
- File must have .csv extension

### Browser Compatibility
This form works best on modern browsers:
- Chrome/Edge (version 90+)
- Firefox (version 88+)
- Safari (version 14+)

## Support

For questions or issues with the submission form, please open an issue on GitHub or contact the competition organizers.

## License

This submission form is part of the Tree Canopy Detection Competition project.
