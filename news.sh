#!/bin/bash
# activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "GO_1" ]; then
    source GO_1/bin/activate
else
    echo "No virtual environment found. Please create one with: python -m venv venv"
    echo "Then activate it and install dependencies: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# run program
echo "Good morning! Starting personalized news feed."
python main.py