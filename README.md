# Fireworks KYC PoV Solution

This repository contains a Python-based Proof of Value (PoV) solution for automating identity verification using Fireworks AI. The solution processes identity documents (e.g., passports and driver's licenses) to extract key information like name, date of birth, and document number, ensuring accuracy and scalability for KYC workflows.

---

## Features
- Utilizes Fireworks AI’s `phi-3` vision-language model for accurate data extraction.
- Includes preprocessing steps like image quality validation for clarity, centering, and visibility.
- Adds logical checks for date reasonability (e.g., ages between 18-100, no future dates).
- Outputs confidence scores and validation results for enhanced transparency.

---

## Installation

### Prerequisites
- Python 3.10 or later
- A Fireworks AI account and API token

### Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/mkessler21/fireworks-kyc-pov.git
   cd fireworks-kyc-pov
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Add your Fireworks API token to an .env file in the project directory:
   ```bash
   FIREWORKS_API_TOKEN=your_api_token_here
### Usage

Run the script to process identity documents:
   ```bash
   python fde_poc.py
```
### Input

 Documents should be placed in the documents folder. Supported formats include .png, .jpeg, and .pdf.

## Output

 The script generates a JSON output containing extracted fields, confidence scores, and a validation result.

# Example Output
   ```bash

{
  "document_type": "Passport",
  "fields": {
    "Full Name": "John Doe",
    "Date of Birth": "15-Mar-1996",
    "Document Number": "963545637",
    "Expiry Date": "14-Apr-2027"
  },
  "confidence_scores": {
    "Full Name": 0.98,
    "Date of Birth": 0.95,
    "Document Number": 0.97,
    "Expiry Date": 0.99
  },
  "validation_result": "Passed"
}
