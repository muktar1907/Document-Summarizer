from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage, vision, firestore
from google.api_core import exceptions as gcp_exceptions
import pypdf

import os
import io
import time 
import uuid
from datetime import datetime, timedelta

from app.Summarizer import abstractive_summary 

# 1. Initialize FastAPI app
app = FastAPI()

# 2. CORS Configuration (Omitted for brevity)
origins = [
    "http://localhost:3000",
    "http://localhost:5173", 
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GCP_PROJECT_ID = "document-summarizer-476701" 

GCS_BUCKET_NAME = "doc_sum_uploaded"

FIRESTORE_COLLECTION = "summaries"

# Explicitly pass the project ID to the clients
storage_client = storage.Client(project=GCP_PROJECT_ID)
vision_client = vision.ImageAnnotatorClient() 

try:
    db = firestore.Client(project=GCP_PROJECT_ID)
except Exception as e:
    print(f"Warning: Firestore client failed to initialize. Persistence will be disabled. {e}")
    db = None

# Supported file formats validation
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'txt'}

def allowed_file(filename):
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(filename):
    """Helper to get file extension."""
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''

def get_document_text(document_id: str, file_ext: str) -> str:
    """
    Retrieves the document from GCS and performs text extraction.
    Prioritizes direct text extraction for PDFs and TXT files.
    Falls back to Cloud Vision API (OCR) for images or if direct PDF parsing fails.
    """
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(document_id)
        
        if not blob.exists():
            raise FileNotFoundError(f"Document not found at path: {document_id}")

        # Download the file contents into an in-memory buffer
        downloaded_contents = io.BytesIO()
        blob.download_to_file(downloaded_contents)
        downloaded_contents.seek(0)
        
        text_content = ""

        # 1. Handle plain text files directly
        if file_ext == 'txt':
            text_content = downloaded_contents.getvalue().decode('utf-8', errors='ignore')
            print("Detected file type: TXT. Reading content directly.")
            
        # 2. Handle digital PDFs directly using pypdf
        elif file_ext == 'pdf':
            print("Detected file type: PDF. Attempting direct text extraction...")
            try:
                reader = pypdf.PdfReader(downloaded_contents)
                for page in reader.pages:
                    # Concatenate extracted text, ensuring empty pages don't cause issues
                    text_content += page.extract_text() or ""
            except Exception as e:
                print(f"pypdf failed: {e}. Falling back to Cloud Vision OCR.")
                text_content = "" # Ensure text_content is empty to trigger OCR fallback
        
        # 3. Fallback to OCR for images, failed PDF parsing, or other binary docs
        if not text_content:
            print(f"Falling back to Cloud Vision OCR for {file_ext}...")
            
            # The PDF must be re-read if pypdf failed and the buffer position was changed
            downloaded_contents.seek(0)
            
            # Prepare image for Vision API (used for images and document OCR)
            image = vision.Image(content=downloaded_contents.getvalue())
            
            # Use DOCUMENT_TEXT_DETECTION for higher fidelity document OCR
            response = vision_client.document_text_detection(image=image)
            
            text_content = response.full_text_annotation.text if response.full_text_annotation else ""
            
            if not text_content:
                print("Warning: OCR returned no text. Trying basic text detection.")
                response = vision_client.text_detection(image=image)
                # The first annotation contains the entire extracted text
                text_content = response.text_annotations[0].description if response.text_annotations else ""

        if not text_content:
            print(f"OCR failed to extract any text from {document_id}")
            raise ValueError("OCR failed to extract readable text from the document.")

        return text_content

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"OCR/GCS Download Error: {e}")
        # Re-raise as HTTPException for client feedback
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e.__class__.__name__}")



# --- Endpoint 1: UPLOAD DOCUMENT ---
@app.post('/upload-document')
async def upload_document(
    document: UploadFile = File(...),
    user_id: str = Form("guest_user")
):
    if not allowed_file(document.filename):
        # Error handling for unsupported formats (as per the proposal)
        raise HTTPException(status_code=415, detail="Unsupported file format.") 

    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # Create a unique blob name using a timestamp, random bytes, and the original filename
        unique_id = f"{time.time_ns()}_{os.urandom(4).hex()}_{document.filename}"
        # Store files in a subfolder named 'raw_documents'
        destination_blob_name = f"raw_documents/{unique_id}"
        blob = bucket.blob(destination_blob_name)
        
        # Read the file contents and upload directly
        contents = await document.read()
        # Explicitly setting content_type is good practice
        blob.upload_from_string(contents, content_type=document.content_type) 
        
        if db:
            # Log metadata to Firestore
            doc_ref = db.collection(FIRESTORE_COLLECTION).document(unique_id)
            doc_ref.set({
                "documentId": destination_blob_name,
                "filename": document.filename,
                "userId": user_id,
                "upload_time": datetime.utcnow(),
                "status": "uploaded",
                "summary": None
            })
            
        # Return the unique path/ID for the frontend to use in the next step
        return JSONResponse(
            content={"message": "File uploaded successfully", "documentId": destination_blob_name},
            status_code=200
        )
        

    except gcp_exceptions.NotFound:
        # Note: This NotFound might still indicate an issue if the GCS bucket itself is not in the correct project.
        raise HTTPException(status_code=500, detail="GCS Bucket not found. Check configuration and authentication.")
    except Exception as e:
        print(f"FATAL GCS/Server Error during upload: {e}") 
        raise HTTPException(status_code=500, detail="Internal server error during upload.")


# --- Endpoint 2: SUMMARIZE (The AI Step) ---
@app.post('/summarize')
async def summarize_document_by_id(
    document_id: str = Form(...),
    length_mode: str = Form("medium"),
):
    if not document_id:
        raise HTTPException(status_code=400, detail="Missing document ID for summarization.")

    try:
        # 1. Determine file extension from the ID
        original_filename = document_id.split('_')[-1]
        file_ext = get_file_extension(original_filename)

        # 2. Extract Text using OCR/Direct Read
        text_content = get_document_text(document_id, file_ext)

        if not text_content or len(text_content.split()) < 20:
             raise HTTPException(status_code=400, detail="Text content is too short or failed to extract for summarization.")

        # 3. Call the Summarizer Function
        summary = abstractive_summary(text_content, length_mode)
        
        # NOTE: This is where you would integrate Firestore (as per the proposal).
        
        if db:
            #document_id passed here is the GCS path "raw_documents/unique_id"
            docs = db.collection(FIRESTORE_COLLECTION).where("documentId", "==", document_id).stream()
            for doc in docs:
                doc.reference.update({
                    "status": "completed",
                    "summary": summary,
                    "length_mode": length_mode,
                    "processed_time": datetime.utcnow()
                })
                
        return JSONResponse(
            content={
                "message": "Document summarized successfully",
                "summary": summary,
                "documentName": original_filename
            },
            status_code=200
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions from get_document_text
        raise
    except Exception as e:
        print(f"FATAL Summarization Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal summarization error: {e.__class__.__name__}")
  
    
@app.get('/history/{user_id}')
async def get_user_history(user_id: str):
    if not db:
        return {"history": []}
    try:
        #Get documents for user, ordered by upload time descending
        docs = db.collection(FIRESTORE_COLLECTION)\
            .where("userId", "==", user_id)\
            .order_by("upload_time", direction=firestore.Query.DESCENDING)\
            .stream()
        
        history = []
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        for doc in docs:
            data = doc.to_dict()
            
            try:
                blob_path = data.get("document_id")
                if blob_path:
                    blob = bucket.blob(blob_path)
                    url = blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(minutes=15),
                        method="GET",
                        response_disposition=f'attachment; filename="{data.get("filename", "document")}"'
                    )
            except Exception as e:
                print(f"Error generating signed URL for {data.get('filename')}: {e}")
                data["download_url"] = None
            
            
            history.append(data)
            
        return {"history": history}
    except Exception as e:
        print(f"Error retrieving user history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving history.")
