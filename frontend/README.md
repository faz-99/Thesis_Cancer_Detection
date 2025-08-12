# Cancer Detection UI

Interactive Vue.js interface for breast cancer detection using histopathological images.

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

## Usage

1. Upload a histopathological image
2. Click "Analyze Image" 
3. View prediction results with confidence scores
4. See risk level and medical explanation

## API Integration

The UI connects to the FastAPI backend at `http://localhost:8000`

Make sure your API server is running before using the interface.

## Tech Stack

- Vue.js 3
- Vite
- Tailwind CSS
- Axios