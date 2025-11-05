import React, { useState } from 'react';
import type { ChangeEvent } from 'react';


const DocumentUploader:React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState('');

  // Supported file formats: PNG, JPEG, PDF, DOCX, etc. [cite: 86]
  const supportedFileTypes = "application/pdf, application/msword, application/vnd.openxmlformats-officedocument.wordprocessingml.document, image/png, image/jpeg, text/plain";
//{ target: { files: string | any[]; }; }
  const handleFileChange = (event: ChangeEvent<HTMLInputElement> ) => {
    // Only allow one file for now
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
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

    // The React UI sends the request to the Python backend [cite: 54]
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
    } catch (error: any) {
      setUploadStatus(`Network error during upload: ${error.message}`);
    }
  };

  return (
    <div className="uploader-container">
      <input
        type="file"
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

export default DocumentUploader;