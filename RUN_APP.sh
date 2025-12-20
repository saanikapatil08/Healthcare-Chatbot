#!/bin/bash
# Quick script to run the Medical Chatbot

cd /Users/saanika/UMDCodingJourney/Healthcare-Chatbot
source .venv/bin/activate
echo "Starting Medical Chatbot..."
echo "Opening browser at http://localhost:8501"
streamlit run app.py
