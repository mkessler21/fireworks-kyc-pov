import os
import base64
import time
from typing import Dict
import fireworks.client
from datetime import datetime
from dotenv import load_dotenv
from enum import Enum
from typing import Dict, List, Optional
import json
import requests
from pydantic import BaseModel, Field

load_dotenv()
fireworks_api_key = os.getenv('FIREWORKS_API_KEY')

class DocumentType(Enum):
    PASSPORT = "passport"
    LICENSE = "license"
    ## other document types
    NATIONAL_ID = "national_id"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"

class ImageQualityCheck(BaseModel):
    centered: str = Field(description="Whether the document is properly centered (yes/no)")
    clear: str = Field(description="Whether the image is clear and readable (yes/no)")
    fully_visible: str = Field(description="Whether the entire document is visible (yes/no)")

class DocumentInfo(BaseModel):
    full_name: str = Field(description="Full name on the document")
    date_of_birth: str = Field(description="Date of birth in YYYY-MM-DD format")
    document_number: str = Field(description="Document number (passport or license number)")
    expiry_date: str = Field(description="Document expiry date in YYYY-MM-DD format")
    document_type: str = Field(description="Type of document (passport/license)")

class DocumentReader:
    def __init__(self):
        self._document_configs = {
            DocumentType.PASSPORT.value: {
                "required_fields": ["full_name", "date_of_birth", "passport_number", "nationality", "expiry_date"],
                "prompt": """Extract in JSON format:
                            - full_name
                            - date_of_birth
                            - passport_number
                            - nationality
                            - expiry_date"""
            },
            DocumentType.LICENSE.value: {
                "required_fields": ["full_name", "date_of_birth", "license_number", "state", "expiry_date"],
                "prompt": """Extract in JSON format:
                            - full_name
                            - date_of_birth
                            - license_number
                            - state
                            - expiry_date"""
            },
            DocumentType.NATIONAL_ID.value: {
                "required_fields": ["full_name", "date_of_birth", "id_number", "nationality"],
                "prompt": """Extract in JSON format:
                            - full_name
                            - date_of_birth
                            - id_number
                            - nationality"""
            },
            DocumentType.UTILITY_BILL.value: {
                "required_fields": ["service_provider", "customer_name", "address", "bill_date", "amount"],
                "prompt": """Extract in JSON format:
                            - service_provider
                            - customer_name
                            - address
                            - bill_date
                            - amount"""
            },
            DocumentType.BANK_STATEMENT.value: {
                "required_fields": ["bank_name", "account_holder", "account_number", "statement_period", "balance"],
                "prompt": """Extract in JSON format:
                            - bank_name
                            - account_holder
                            - account_number
                            - statement_period
                            - balance"""
            }
        }

    def get_document_prompt(self, doc_type: str) -> Optional[str]:
        """Get the extraction prompt for a specific document type"""
        config = self._document_configs.get(doc_type)
        return config["prompt"] if config else None

    def get_required_fields(self, doc_type: str) -> List[str]:
        """Get required fields for a specific document type"""
        config = self._document_configs.get(doc_type)
        return config["required_fields"] if config else []

    def validate_extracted_data(self, doc_type: str, extracted_data: Dict) -> Dict:
        """Validate extracted data against required fields"""
        required_fields = self.get_required_fields(doc_type)
        if not required_fields:
            return {
                "is_valid": False,
                "error": f"Unknown document type: {doc_type}"
            }

        try:
            # If extracted_data is a string (JSON), parse it
            if isinstance(extracted_data, str):
                data = json.loads(extracted_data)
            else:
                data = extracted_data

            missing_fields = [field for field in required_fields if field not in data]
            
            return {
                "is_valid": len(missing_fields) == 0,
                "missing_fields": missing_fields,
                "validation_details": {
                    "required_fields": required_fields,
                    "provided_fields": list(data.keys())
                }
            }
        except json.JSONDecodeError:
            return {
                "is_valid": False,
                "error": "Invalid JSON format in extracted data"
            }

class KYCProcessor:
    def __init__(self, api_key: str):
        """Sets up the KYC processor with API credentials and initializes the document reader"""
        self.api_key = api_key
        fireworks.client.api_key = api_key
        self.model = "accounts/fireworks/models/phi-3-vision-128k-instruct"
        self.document_reader = DocumentReader()
        
    def process_kyc_document(self, image_path: str) -> Dict:
        """
        Main KYC processing pipeline with performance metrics
        """
        start_time = time.time()
        
        try:
            print("Starting document processing...")
            
            # Load and encode image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                if len(image_data) > 20 * 1024 * 1024:
                    raise ValueError("Image file is too large. Please use an image smaller than 20MB")
                
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                image_url = f"data:image/png;base64,{image_base64}"
            
            # Step 1: Image Quality Check
            quality_check = self._validate_image_quality(image_url)
            if not all(v == "yes" for v in quality_check.values()):
                return {
                    "status": "error",
                    "error_message": "Image quality check failed",
                    "quality_issues": quality_check,
                    "processing_time_seconds": round(time.time() - start_time, 2)
                }
            
            # Step 2: Document Classification
            doc_type = self._detect_document_type(image_url)
            
            # Step 3: Information Extraction
            extracted_info = self._extract_document_info(image_url, doc_type)
            
            # Step 4: Date Validation
            try:
                extracted_data = json.loads(extracted_info["extracted_data"])
                date_validation = self._validate_date_reasonability(extracted_data.get("date_of_birth", ""))
                if not date_validation["is_reasonable"]:
                    return {
                        "status": "error",
                        "error_message": "Date validation failed",
                        "date_issues": date_validation["issues"],
                        "processing_time_seconds": round(time.time() - start_time, 2)
                    }
            except json.JSONDecodeError:
                pass  # Handle in general validation
            
            # Step 5: General Validation
            validation_result = self._validate_extracted_info(extracted_info, doc_type)
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "document_type": doc_type,
                "extracted_info": extracted_info,
                "validation_result": validation_result,
                "quality_check": quality_check,
                "processing_metrics": {
                    "processing_time_seconds": round(processing_time, 2),
                    "confidence_score": extracted_info.get("confidence_score", 0),
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "processing_time_seconds": round(time.time() - start_time, 2)
            }

    def _detect_document_type(self, image_base64: str) -> str:
        """
        Uses the vision model to analyze the image and determine if it's a passport
        or driver's license. Returns a simple string response of 'passport' or 'license'.
        """
        response = fireworks.client.ChatCompletion.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Is this image a passport or a driver's license? Respond with only 'passport' or 'license'.",
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                }],
            }]
        )
        return response.choices[0].message.content.strip().lower()

    def _extract_document_info(self, image_base64: str, doc_type: str) -> Dict:
        """
        Extracts relevant information from the document based on its type.
        For passports: extracts name, DOB, passport number, nationality, and expiry
        For licenses: extracts name, DOB, license number, state, and expiry
        
        Returns a dictionary with the extracted data, confidence score, and validation results
        """
        prompt = self.document_reader.get_document_prompt(doc_type)
        if not prompt:
            raise ValueError(f"Unsupported document type: {doc_type}")
        
        response = fireworks.client.ChatCompletion.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt,
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                }],
            }]
        )
        
        extracted_data = response.choices[0].message.content
        validation_result = self.document_reader.validate_extracted_data(doc_type, extracted_data)
        
        return {
            "extracted_data": extracted_data,
            "confidence_score": 0.92,  # Mock confidence score for PoC
            "validation_result": validation_result
        }

    def _validate_extracted_info(self, extracted_info: Dict, doc_type: str) -> Dict:
        """
        Basic validation of extracted information
        """
        # For PoC, implementing basic validation
        required_fields = {
            "passport": ["full_name", "date_of_birth", "passport_number"],
            "license": ["full_name", "date_of_birth", "license_number"]
        }
        
        # Mock validation for PoC
        return {
            "is_valid": True,
            "missing_fields": [],
            "compliance_check": "passed"
        }

    def _validate_image_quality(self, image_url: str) -> Dict:
        """
        Check image quality and document positioning using JSON mode
        """
        try:
            response = fireworks.client.ChatCompletion.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": "Analyze this image and check document positioning and quality. Respond in JSON format."
                }, {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }],
                response_format={
                    "type": "json_object",
                    "schema": ImageQualityCheck.model_json_schema()
                }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in image quality check: {str(e)}")
            return {"centered": "no", "clear": "no", "fully_visible": "no"}

    def _validate_date_reasonability(self, date_str: str) -> Dict:
        """
        Validate if the date is reasonable for an active ID
        """
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now()
            
            age = today.year - date.year - ((today.month, today.day) < (date.month, date.day))
            
            return {
                "is_reasonable": 18 <= age <= 100,
                "issues": [
                    "Age below 18" if age < 18 else None,
                    "Age appears unreasonable (over 100 years)" if age > 100 else None,
                    "Date is in the future" if date > today else None
                ]
            }
        except ValueError:
            return {
                "is_reasonable": False,
                "issues": ["Invalid date format"]
            }

def main():
    api_key = os.getenv("FIREWORKS_API_KEY")
    print(f"Using API key: {api_key[:8]}..." if api_key else "No API key found!")
    
    # Process sample documents
    processor = KYCProcessor(api_key)
    
    # Example processing
    result = processor.process_kyc_document("documents/License 1.png")
    print("\nProcessing Results:")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Document Type: {result['document_type']}")
        print(f"Processing Time: {result['processing_metrics']['processing_time_seconds']}s")
        print(f"Confidence Score: {result['processing_metrics']['confidence_score']}")

if __name__ == "__main__":
    main()