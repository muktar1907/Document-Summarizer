from fastapi import FastAPI;
from flask import Flask, request, jsonify
from google.cloud import storage
import os

app = FastAPI()
uploadapp = Flask(__name__)

@app.get("/api/hello")
def hello():
    return {"message": "Hello from FastAPI"}

# Configuration
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'your-gcs-bucket-name')
storage_client = storage.Client()

# Supported file formats validation (MIME types)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@uploadapp.route('/upload-document', methods=['POST'])
def upload_document():
    # 1. Check for file in request
    if 'document' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    file = request.files['document']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    # 2. File Format Validation/Error Handling 
    if not allowed_file(file.filename):
        return jsonify({"message": "Unsupported file format."}), 415

    # 3. Securely upload to Google Cloud Storage 
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        # Use a unique name to prevent collisions
        destination_blob_name = f"raw_documents/{file.filename}_{os.urandom(8).hex()}"
        blob = bucket.blob(destination_blob_name)
        
        # Upload the file's content
        blob.upload_from_file(file)

        # 4. Store metadata in Google Cloud Firestore (NoSQL database)
        # In a full implementation, you'd add code here to store the document's metadata
        # (e.g., user ID, GCS path, status) in Firestore.
        
        document_id = destination_blob_name.split('/')[-1]
        
        # The data is transported to the Google Cloud Storage container
        return jsonify({"message": "File uploaded successfully", "documentId": document_id}), 200

    except Exception as e:
        print(f"GCS Upload Error: {e}")
        return jsonify({"message": "Internal server error during upload."}), 500

if __name__ == '__main__':
    # Used for local testing, Cloud Run/Functions will use different entry points
    app.run(debug=True, port=8080)