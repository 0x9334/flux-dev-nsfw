#!/bin/bash
echo "Installing remaining dependencies..."

# Install other dependencies from requirements.txt
pip install git+https://github.com/huggingface/diffusers loguru mmgp protobuf sentencepiece transformers==4.52.4 accelerate==1.7.0 datasets python-dotenv fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 huggingface_hub>=0.19.0 pillow>=10.1.0 numpy>=1.24.0 requests>=2.31.0 python-dotenv>=1.0.0

echo "Installation complete!" 