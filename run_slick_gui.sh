#!/bin/bash
# Launch Script for GeoNex Slick GUI v3.0

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
if [ -d "$DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$DIR/venv/bin/activate"
else
    echo "Error: Virtual environment not found in $DIR/venv"
    exit 1
fi

# Run the Slick GUI
echo "Starting GeoNex Slick GUI v3.0..."
python3 "$DIR/slick_gui.py"
