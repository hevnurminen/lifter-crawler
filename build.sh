#!/usr/bin/env bash
# Install system dependencies
apt-get update && apt-get install -y python3-dev build-essential

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt 