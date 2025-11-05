from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage

import os
import io # Used to read file content into memory for upload

# 1. Initialize FastAPI app only
app = FastAPI()

# 2. CORS Configuration (Copied from your original file)
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

# 3. Google Cloud Configuration
GCS_BUCKET_NAME = "doc_sum_uploaded"
storage_client = storage.Client()

# Supported file formats validation
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'txt'}

def allowed_file(filename):
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 4. Convert the upload route to FastAPI syntax
@app.post('/upload-document')
async def upload_document(document: UploadFile = File(...)): # FastAPI way to handle file upload
    # 1. Check for file and filename
    if not document.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    # 2. File Format Validation/Error Handling 
    if not allowed_file(document.filename):
        # Use HTTPException for standardized error response
        raise HTTPException(status_code=415, detail="Unsupported file format.")

    # 3. Securely upload to Google Cloud Storage 
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # Use a unique name
        unique_id = os.urandom(8).hex()
        destination_blob_name = f"raw_documents/{document.filename}_{unique_id}"
        blob = bucket.blob(destination_blob_name)
        
        # FastAPI's UploadFile requires reading the content before uploading
        contents = await document.read()
        
        # Upload from the in-memory buffer
        blob.upload_from_file(io.BytesIO(contents)) 

        # 4. Store metadata (placeholder)
        document_id = destination_blob_name.split('/')[-1]
        
        # Successful response MUST be a JSONResponse
        return JSONResponse(
            content={"message": "File uploaded successfully", "documentId": document_id},
            status_code=200
        )
        

    except Exception as e:
        # Detailed error logging for debugging
        print(f"FATAL GCS/Server Error: {e}") 
        # Return a standardized error to the client
        raise HTTPException(status_code=500, detail="Internal server error during upload.")


# You can keep your test route
@app.get("/api/hello")
def hello():
    return {"message": "Hello from FastAPI"}