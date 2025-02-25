# OCR Service API

A fun OCR I made using express, sharp, and tesseract!

## Features

- Text extraction from images using Tesseract.js
- Support for multiple image input formats (URL, base64, bytes)
- Batch processing capability
- Performance metrics for each operation
- Image preprocessing with Sharp
- Domain allowlist for URL-based requests
- Comprehensive error handling and validation

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd ocr-service
```

2. Install dependencies:
```bash
npm install
```

e. Start the server:
```bash
npm start
```

## API Endpoints

### 1. Single Image Recognition
`POST /recognize`

Extracts text from a single image.

#### Request Body Format:

URL-based:
```json
{
  "url": "https://example.com/image.jpg"
}
```

Base64-based:
```json
{
  "base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

Bytes-based:
```json
{
  "bytes": [255, 216, 255, ...]
}
```

#### Query Parameters:

- `mode`: (optional) "fast" or "accurate"
- `languages`: (optional) Array of language codes (default: "eng")
- `detect_language`: (optional) Boolean
- `autocorrect`: (optional) Boolean

#### Response Format:

Success:
```json
{
  "success": true,
  "data": {
    "width": 800,
    "height": 600,
    "content": "Extracted text content",
    "confidence": 98.5,
    "processingTimeMs": 1234
  }
}
```

Error:
```json
{
  "success": false,
  "error": "Error message",
  "processingTimeMs": 123
}
```

### 2. Batch Processing
`POST /batch`

Process multiple images in parallel.

#### Request Body Format:
```json
[
  {
    "url": "https://example.com/image1.jpg"
  },
  {
    "base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }
]
```

#### Response Format:

Success:
```json
{
  "success": true,
  "data": [
    {
      "success": true,
      "data": {
        "width": 800,
        "height": 600,
        "content": "Extracted text content",
        "confidence": 98.5,
        "processingTimeMs": 1234
      }
    },
    // ... more results
  ],
  "totalProcessingTimeMs": 2345
}
```

## Allowed Domains

For URL-based requests, images must be hosted on one of these domains:
- discord.mx

## Example Usage

### Using curl

Single image recognition:
```bash
curl -X POST http://localhost:3000/recognize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://i.imgur.com/example.jpg"}'
```

Batch processing:
```bash
curl -X POST http://localhost:3000/batch \
  -H "Content-Type: application/json" \
  -d '[{"url": "https://i.imgur.com/example1.jpg"}, {"url": "https://i.imgur.com/example2.jpg"}]'
```

### Using JavaScript/TypeScript

```typescript
// Single image recognition
const response = await fetch('http://localhost:3000/recognize', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    url: 'https://i.imgur.com/example.jpg'
  })
});

const result = await response.json();

// Batch processing
const batchResponse = await fetch('http://localhost:3000/batch', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify([
    { url: 'https://i.imgur.com/example1.jpg' },
    { url: 'https://i.imgur.com/example2.jpg' }
  ])
});

const batchResult = await batchResponse.json();
```

## Self-Hosting Guide

1. Server Requirements:
   - 1GB RAM minimum (2GB recommended)
   - 1 CPU core minimum (2+ recommended)
   - 1GB free disk space

2. Setup Steps:
   ```bash
   # Install Node.js
   curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
   sudo apt-get install -y nodejs

   # Clone repository
   git clone <your-repository-url>
   cd ocr-service

   # Install dependencies
   npm install

   # Start with PM2 (recommended for production)
   npm install -g pm2
   pm2 start npm --name "ocr-service" -- start
   ```

3. Using Docker:
   ```bash
   # Build image
   docker build -t ocr-service .

   # Run container
   docker run -p 3000:3000 ocr-service
   ```

## Error Handling

The service provides detailed error messages for common issues:
- Invalid image format
- Domain not allowed
- Network errors
- Invalid JSON format
- Validation errors

## Performance Considerations

- The service processes batch requests in parallel
- Each request includes processing time metrics
- Image preprocessing is optimized using Sharp
- Memory usage is optimized for large images

## Security Notes

- URL requests are restricted to allowed domains
- Image size is limited to 24MB
- Input validation is performed on all requests
- Error messages are sanitized