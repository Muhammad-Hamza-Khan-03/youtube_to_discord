#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Initialize environment variables from template
if [ ! -f .env ]; then
    cp .env.template .env
    echo ".env file created from .env.template. Please update it with your API keys."
else
    echo ".env file already exists. Skipping copy."
fi

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
