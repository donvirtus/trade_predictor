#!/bin/bash

# Auto-activate conda environment for VSCode terminal
# This script will be executed when VSCode opens a new terminal

# Load conda initialization
if [ -f "/home/donvirtus/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/donvirtus/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/donvirtus/miniconda3/bin/activate" ]; then
    source "/home/donvirtus/miniconda3/bin/activate"
fi

# Activate projects environment
conda activate projects 2>/dev/null || echo "Warning: Could not activate 'projects' environment"

# Show environment info
echo "🐍 Python Environment: $(which python)"
echo "📍 Working Directory: $(pwd)"
echo "✅ Ready to run scripts!"

# Load default bashrc for other settings
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi