import { useState, type ChangeEvent } from 'react'
import './App.css'


const DocumentUploader = () => {
  // Best Practice: Explicitly type the state for selectedFile as File or null
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState('');

  // Supported file formats: PNG, JPEG, PDF, DOCX, etc.
  const supportedFileTypes = "application/pdf, application/msword, application/vnd.openxmlformats-officedocument.wordprocessingml.document, image/png, image/jpeg, text/plain";

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    // FIX: Assign event.target.files to a variable for safe access
    const files = event.target.files;

    // Now TypeScript is happy: check if files is not null AND has items
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first.');
      return;
    }

    setUploadStatus('Uploading...');
    const formData = new FormData();
    formData.append('document', selectedFile);

// The React UI sends the request to the Python backend
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/upload-document`, {
        method: 'POST',
        // The browser automatically sets the 'Content-Type' header for FormData
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setUploadStatus(`Upload successful! Document ID: ${data.documentId}`);
        // Optionally, clear the file input after success
        setSelectedFile(null);
      } else {
        const errorData = await response.json();
        setUploadStatus(`Upload failed: ${errorData.message}`);
      }
    } catch (error: unknown) {
      let errorMessage = "Unknown error";
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      else if (typeof error === 'object' && error !== null && 'message' in error) {
        errorMessage = `Network error during upload: ${(error as {message: string }).message}`;
      }
      setUploadStatus(errorMessage);
    }
  };

  return (
    <div className="uploader-container">
      <label htmlFor = "document-upload">Select a document to upload:</label>
      <input
        type="file"
        id = "document-upload"
        accept={supportedFileTypes}
        onChange={handleFileChange}
      />
      <button onClick={handleUpload} disabled={!selectedFile || uploadStatus === 'Uploading...'}>
        {uploadStatus === 'Uploading...' ? 'Uploading...' : 'Upload Document'}
      </button>
      <p>{uploadStatus}</p>
    </div>
  );
};

export default DocumentUploader ;