#!/bin/bash

# Quick script to run the Vajra Prototype in Docker locally

echo "🚀 Starting Vajra Prototype in Docker..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOF
# Copy your API keys here
OPENAI_API_KEY=your-openai-key-here
OPENWEATHER_API_KEY=your-openweather-key-here
GOOGLE_PLACES_API_KEY=your-google-places-key-here
EOF
    echo "✅ Created .env file. Please edit it with your API keys."
    exit 1
fi

# Build the image
echo "📦 Building Docker image..."
docker build -t vajra-prototype .

# Run the container
echo "🏃 Starting container..."
docker run -p 8501:8501 \
  --env-file .env \
  --name vajra-prototype \
  --rm \
  vajra-prototype

echo "✅ App should be running at http://localhost:8501"

