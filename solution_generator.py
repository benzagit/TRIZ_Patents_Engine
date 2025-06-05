import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import requests
import random
import csv
from io import StringIO
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import logging
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception as e:
    st.error(f"Error loading spaCy model: {e}")
    nlp = None

# NVIDIA Nemotron 70B API Setup
API_KEY = ""  # Replace with your actual API key
BASE_URL = "https://integrate.api.nvidia.com/v1"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def generate_response(prompt):
    logger.debug(f"Generating response for prompt: {prompt[:100]}...")
    payload = {
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        logger.debug(f"Response generated: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Error in API call: {e}")
        st.error(f"Error in API call: {e}")
        return ""

# TRIZ Setup
principles = [
    "Segmentation", "Taking out", "Local quality", "Asymmetry", "Merging",
    "Universality", "Nested doll", "Anti-weight", "Preliminary anti-action", "Preliminary action",
    "Beforehand cushioning", "Equipotentiality", "The other way round", "Spheroidality", "Dynamics",
    "Partial or excessive actions", "Another dimension", "Mechanical vibration", "Periodic action", "Continuity of useful action",
    "Skipping", "Blessing in disguise", "Feedback", "Intermediary", "Self-service",
    "Copying", "Cheap short-living objects", "Mechanics substitution", "Pneumatics and hydraulics", "Flexible shells and thin films",
    "Porous materials", "Color changes", "Homogeneity", "Discarding and recovering", "Parameter changes",
    "Phase transitions", "Thermal expansion", "Strong oxidants", "Inert atmosphere", "Composite materials"
]

TRIZ_PARAMS = {i+1: param for i, param in enumerate([
    "Weight of moving object", "Weight of stationary object", "Length of moving object", "Length of stationary object",
    "Area of moving object", "Area of stationary object", "Volume of moving object", "Volume of stationary object",
    "Speed", "Force", "Stress, pressure", "Shape", "Stability", "Strength", "Durability of moving object",
    "Durability of stationary object", "Temperature", "Brightness", "Energy consumption of moving object",
    "Energy consumption of stationary object", "Power", "Loss of energy", "Loss of substance", "Loss of information",
    "Loss of time", "Quantity of substance", "Reliability", "Measurement accuracy", "Manufacturing precision",
    "Harmful factors affecting the object", "Harmful factors induced by the object", "Ease of manufacture", "Ease of use",
    "Repairability", "Adaptability", "Complexity of product", "Complexity of control", "Degree of automation", "Productivity"
])}
TRIZ_MATRIX = [
    ["*", [], [15, 8, 29, 34], [], [29, 17, 38, 34], [], [29, 2, 40, 28], [], [2, 8, 15, 38], [8, 10, 18, 37], [10, 36, 37, 40], [10, 14, 35, 40], [1, 35, 19, 39], [28, 27, 18, 40], [5, 34, 31, 35], [], [6, 29, 4, 38], [], [35, 12, 34, 31], [], [12, 36, 18, 31], [6, 2, 34, 19], [5, 35, 3, 31], [10, 24, 35], [10, 35, 20, 28], [3, 26, 18, 31], [1, 3, 11, 27], [28, 27, 35, 26], [28, 35, 26, 18], [22, 21, 18, 27], [22, 35, 31, 39], [27, 28, 1, 36], [35, 3, 2, 24], [2, 27, 28, 11], [29, 5, 15, 8], [26, 30, 36, 34], [28, 29, 26, 32], [26, 35, 18, 19], [35, 3, 24, 37]],
    [[], "*", [], [10, 1, 29, 35], [], [35, 30, 13, 2], [], [5, 35, 14, 2], [], [8, 10, 19, 35], [13, 29, 10, 18], [13, 10, 29, 14], [26, 39, 1, 40], [28, 2, 10, 27], [], [2, 27, 19, 6], [28, 19, 32, 22], [], [], [18, 19, 28, 1], [15, 19, 18, 22], [18, 19, 28, 15], [5, 8, 13, 30], [10, 15, 35], [10, 20, 35, 26], [19, 6, 18, 26], [10, 28, 8, 3], [18, 26, 28], [10, 1, 35, 17], [2, 19, 22, 37], [35, 22, 1, 39], [], [6, 13, 1, 32], [2, 27, 28, 11], [19, 15, 29], [1, 10, 26, 39], [25, 28, 17, 15], [2, 26, 35], [1, 28, 15, 35]],
    [[8, 15, 29, 34], [], "*", [], [15, 17, 4], [], [7, 17, 4, 35], [], [13], [4, 17], [], [1, 8, 10, 29], [1, 8, 15, 34], [8, 35, 29, 34], [19], [], [10, 15, 19], [32], [8, 35, 24], [], [1, 35], [7, 2, 35, 39], [4, 29, 23, 10], [1, 24], [], [29, 35], [10, 14, 29, 40], [28, 32, 4], [10, 28, 29, 37], [1, 15, 17, 24], [17, 15], [1, 29, 17], [15, 29, 35, 4], [1, 28, 10], [14, 15, 1, 16], [1, 19, 26, 24], [35, 1, 26, 24], [17, 24, 26, 16], [14, 4, 28, 29]],
    [[], [35, 28, 40, 29], [], "*", [], [17, 7, 10, 40], [], [35, 8, 2, 14], [], [24, 28], [1, 14, 35], [13, 14, 15, 7], [39, 37, 35], [15, 14, 28, 26], [], [], [3, 35, 38, 18], [3, 25], [], [], [24], [6, 28], [10, 28, 24, 35], [24, 26], [30, 29, 14], [], [15, 29, 28], [32, 28, 3], [23, 2, 10], [1, 18], [], [15, 17, 27], [2, 25], [3], [1, 35], [1, 26], [26], [], [30, 14, 7, 26]],
    [[2, 17, 29, 4], [], [14, 15, 18, 4], [], "*", [], [7, 14, 17, 4], [], [29, 30, 4, 34], [19, 30, 35, 2], [10, 15, 36, 28], [5, 34, 29, 4], [11, 2, 13, 39], [3, 15, 40, 14], [24], [], [2, 15, 16], [15, 32, 19, 13], [19, 32], [], [19, 10, 32, 18], [15, 17, 30, 26], [10, 35, 2, 39], [30, 26], [], [29, 30, 6, 13], [], [26, 28, 32, 3], [23, 2], [22, 33, 28, 1], [17, 2, 18, 39], [13, 1, 26, 24], [15, 17, 13, 16], [15, 13, 10, 1], [15, 30], [13], [2, 36, 26, 18], [14, 30, 28, 23], [10, 26, 34, 2]],
    [[], [30, 2, 14, 18], [], [26, 7, 9, 39], [], "*", [], [], [], [1, 18, 35, 36], [10, 15, 36, 37], [], [2, 38], [40], [], [2, 10, 19, 30], [35, 39, 38], [], [], [], [17, 32], [], [10, 14, 18, 39], [30, 16], [10, 35, 4, 18], [2, 18, 40, 4], [32, 35, 40, 4], [26, 28, 32, 3], [2, 29, 18, 36], [27, 2, 39, 35], [], [40, 16], [], [16], [15, 16], [1, 18, 36], [2, 35, 30, 18], [23], [10, 15, 17, 7]],
    [[2, 26, 29, 40], [], [1, 7, 4, 35], [], [1, 7, 4, 17], [], "*", [], [29, 4, 38, 34], [15, 35, 36, 37], [6, 35, 36, 37], [1, 15, 29, 4], [28, 10, 1, 39], [9, 14, 15, 7], [6, 35, 4], [], [34, 39, 10, 18], [2, 13, 10], [35], [], [35, 6, 13, 18], [7, 15, 13, 16], [36, 39, 34, 10], [2, 22], [2, 6, 34, 10], [29, 30, 7], [14, 1, 40, 11], [25, 26, 28], [25, 28, 2, 16], [22, 21, 27, 35], [17, 2, 40, 1], [], [15, 13, 30, 12], [10], [15, 29], [], [29, 26, 4], [35, 34, 16, 24], [10, 6, 2, 34]],
    [[], [35, 10, 19, 14], [19, 14], [35, 8, 2, 14], [], [], [], "*", [], [2, 18, 37], [24, 35], [], [34, 28, 35, 40], [9, 14, 17, 15], [], [35, 34, 38], [35, 6, 4], [], [], [], [], [], [10, 39, 35, 34], [], [35, 16, 32, 18], [35, 3], [2, 35, 16], [], [35, 10, 25], [34, 39, 19, 27], [30, 18, 35, 4], [35], [], [1], [], [13, 1], [2, 17, 26], [], [35, 37, 10, 2]],
    [[2, 28, 13, 38], [], [13, 14, 8], [], [29, 30, 34], [], [7, 29, 34], [], "*", [13, 28, 15, 19], [6, 18, 38, 40], [35, 15, 18, 34], [28, 33, 1, 18], [8, 3, 26, 14], [3, 19, 35, 5], [], [28, 30, 36, 2], [10, 13, 19], [8, 15, 35, 38], [], [19, 35, 38, 2], [14, 20, 19, 35], [10, 13, 28, 38], [13, 26], [], [10, 19, 29, 38], [11, 35, 27, 28], [28, 32, 1, 24], [10, 28, 32, 25], [1, 28, 35, 23], [2, 24, 35, 21], [35, 13, 8, 1], [32, 28, 13, 12], [34, 2, 28, 27], [26], [10, 28, 4, 34], [33, 4, 27, 16], [10, 18], []],
    [[8, 1, 37, 18], [18, 13, 1, 28], [17, 19, 9, 36], [24, 28], [15, 19], [1, 18, 36, 37], [15, 9, 12, 37], [2, 36, 18, 37], [13, 28, 15, 12], "*", [18, 21, 11], [10, 35, 40, 34], [35, 10, 21], [35, 10, 14, 27], [19], [], [35, 10, 21], [], [19, 17, 10], [1, 16, 36, 37], [19, 35, 18, 37], [14, 15], [8, 35, 40, 5], [], [10, 37, 36], [14, 29, 18, 36], [3, 35, 13, 21], [35, 10, 23, 24], [28, 29, 37, 36], [1, 35, 40, 18], [13, 3, 36, 24], [15, 37, 18, 1], [1, 28, 3, 25], [11, 15], [15, 17, 18, 20], [26, 35, 10, 18], [36, 37, 10, 19], [2, 35], [3, 28, 35, 37]],
    [[10, 36, 37, 40], [13, 29, 10, 18], [35, 10, 36], [35, 1, 14, 16], [10, 15, 36, 28], [10, 15, 36, 37], [6, 35, 10], [35, 24], [6, 35, 36], [36, 35, 21], "*", [35, 4, 15, 10], [35, 33, 2, 40], [9, 18, 3, 40], [], [], [35, 39, 19, 2], [], [14, 24, 10, 37], [], [10, 35, 14], [2, 36, 25], [10, 36, 3, 37], [], [37, 36, 4], [10, 14, 36], [10, 13, 19, 35], [6, 28, 25], [3, 35], [], [23, 32, 27, 18], [1, 35, 16], [11], [2], [35], [], [2, 36, 37], [35, 24], [10, 14, 35, 37]],
    [[10, 14, 35, 40], [15, 10, 26, 3], [29, 34, 5, 4], [13, 14, 10, 7], [5, 34, 4, 10], [], [14, 4, 15, 22], [], [35, 15, 34, 18], [35, 10, 37, 40], [34, 15, 10, 14], "*", [33, 1, 18, 4], [30, 14, 10, 40], [14, 26, 9, 25], [], [22, 14, 19, 32], [13, 15, 32], [2, 6, 34, 14], [], [2], [14], [35, 29, 3, 5], [], [14, 10, 34, 17], [36, 22], [10, 40, 16], [28, 32, 1], [32, 30, 40], [22, 1, 2, 35], [35, 1], [13, 2, 17, 28], [32, 15, 26], [2, 13, 1], [1, 15, 29], [16, 29, 1, 28], [15, 13, 39], [], [17, 26, 34, 10]],
    [[1, 35, 19, 39], [26, 39, 1, 40], [13, 15, 1, 28], [37], [13, 2], [39], [28, 10, 19, 39], [34, 28, 35, 40], [33, 15, 28, 18], [10, 35, 21, 16], [2, 35, 40], [22, 1, 18, 4], "*", [15, 17], [13, 27, 10, 35], [39, 3, 35, 23], [35, 1, 32], [32, 3, 27, 16], [13, 19], [27, 4, 29, 18], [32, 35, 27, 31], [14, 2, 39, 6], [21, 4, 30, 40], [], [35, 27], [15, 32, 35], [], [13], [18], [35, 24, 30, 18], [35, 40, 27, 39], [35, 19], [32, 35, 30], [2, 35, 10, 16], [35, 30, 34, 2], [2, 35, 22, 26], [35, 22, 39, 23], [], [23, 35, 40, 3]],
    [[28, 27, 18, 40], [28, 2, 10, 27], [1, 15, 8, 35], [15, 14, 28, 26], [33, 4, 40, 29], [9, 40, 28], [10, 15, 14, 7], [9, 14, 17, 15], [8, 13, 26, 14], [10, 18, 3, 14], [10, 3, 18, 40], [10, 30, 35, 40], [13, 17, 35], "*", [26, 27], [], [19, 40], [35, 19], [19, 35, 10], [35], [10, 26, 35, 28], [35], [35, 28, 31, 40], [], [29, 3, 28, 10], [], [3, 27, 16], [3, 27, 16], [3, 27], [18, 35, 37, 1], [15, 35, 22, 2], [11, 3, 10, 32], [32, 40, 25, 2], [], [32, 15], [21, 3, 25, 28], [27, 3, 15, 40], [15], [29, 35, 10, 14]],
    [[5, 34, 31, 35], [], [2, 19, 9], [], [3, 17, 19], [], [10, 2, 19, 30], [], [3, 35, 5], [16, 19], [], [14, 26, 28, 25], [], [10, 27], "*", [], [19, 35, 39], [2, 19, 4, 35], [28, 6, 35, 18], [], [19, 10, 35, 38], [], [28, 27, 3, 18], [10], [10, 28, 18], [3, 35, 10, 40], [], [3], [3, 27, 16, 40], [22, 15, 33, 28], [21, 39, 16, 22], [], [1, 22, 27], [], [1, 35, 13], [10, 4, 29, 15], [19, 29, 39, 35], [], [35, 17, 14, 19]],
    [[], [6, 27, 19, 16], [], [1, 40, 35], [], [], [], [35, 34, 38], [], [], [], [], [39, 3, 35, 23], [], [], "*", [19, 18, 36, 40], [], [], [], [16], [], [27, 16, 18, 38], [10], [28, 20, 10, 16], [3, 35, 31], [34, 27, 6, 40], [10, 26, 24], [], [17, 1, 40, 33], [22], [35, 10], [1], [1], [2], [], [25, 34, 6, 35], [1], [20, 10, 16, 38]],
    [[6, 29, 4, 38], [28, 19, 32, 22], [10, 15, 19], [3, 35, 38, 18], [2, 15, 16], [2, 10, 19, 30], [34, 39, 10, 18], [35, 6, 4], [28, 30, 36, 2], [35, 10, 21], [35, 39, 19, 2], [22, 14, 19, 32], [35, 1, 32], [19, 40], [19, 35, 39], [19, 18, 36, 40], "*", [32, 30, 21, 16], [19, 15, 3, 17], [], [21, 4, 17, 25], [21, 17, 35, 38], [21, 36, 29, 31], [], [35, 28, 21, 18], [3, 17, 30, 39], [19, 35, 3, 10], [32, 19, 24], [24], [22, 33, 35, 2], [22, 35, 2, 24], [26, 27], [26, 27], [16], [2, 18, 27], [2, 17, 16], [3, 27, 35, 31], [26, 2, 19, 16], [15, 28, 35]],
    [[], [], [32], [3, 25], [15, 32, 19, 13], [], [2, 13, 10], [], [10, 13, 19], [], [], [13, 15, 32], [32, 3, 27, 16], [35, 19], [2, 19, 4, 35], [], [32, 35, 19], "*", [32, 1, 19], [32, 35, 1, 15], [32], [13, 16, 1, 6], [], [], [19, 1, 26, 17], [1, 19], [], [1, 15, 32], [3, 32], [15, 19], [35, 19, 32, 39], [19, 35, 28, 26], [28, 26, 19], [15, 17, 13, 16], [], [6, 32, 13], [32, 15], [2, 26, 10], [2, 25, 16]],
    [[35, 12, 34, 31], [], [8, 35, 24], [], [19, 32], [], [35], [], [8, 15, 35, 38], [19, 17, 10], [14, 24, 10, 37], [2, 6, 34, 14], [13, 19], [19, 35, 10], [28, 6, 35, 18], [], [19, 15, 3, 17], [32, 1, 19], "*", [], [6, 19, 37, 18], [12, 22, 15, 24], [35, 24, 18, 5], [], [35, 38, 19, 18], [34, 23, 16, 18], [19, 21, 11, 27], [], [], [1, 35, 6, 27], [2, 35, 6], [28, 26, 30], [19, 35], [1, 15, 17, 28], [15, 17, 13, 16], [2, 29, 27, 28], [35, 38], [32, 2], [12, 28, 35]],
    [[], [18, 19, 28, 1], [], [], [], [], [], [], [], [1, 16, 36, 37], [], [], [27, 4, 29, 18], [35], [], [], [], [32, 35, 1, 15], [], "*", [], [], [28, 27, 18, 31], [], [], [3, 35, 31], [10, 36, 23], [], [], [10, 2, 22, 37], [19, 22, 18], [], [], [], [], [], [19, 35, 16, 25], [], []],
    [[12, 36, 18, 31], [15, 19, 18, 22], [1, 35], [24], [19, 10, 32, 18], [17, 32], [35, 6, 13, 18], [], [19, 35, 38, 2], [19, 35, 18, 37], [10, 35, 14], [2], [32, 35, 27, 31], [10, 26, 35, 28], [19, 10, 35, 38], [16], [21, 4, 17, 25], [32], [6, 19, 37, 18], [], "*", [10, 35, 38], [28, 27, 18, 38], [10, 19], [35, 20, 10, 6], [4, 34, 19], [19, 24, 26, 31], [32, 15, 2], [32, 2], [19, 22, 31, 2], [2, 35, 18], [], [26, 35, 10], [35, 2, 10, 34], [19, 17, 34], [19, 30, 34], [19, 35, 16], [], [28, 35, 34]],
    [[6, 2, 34, 19], [18, 19, 28, 15], [7, 2, 35, 39], [6, 28], [15, 17, 30, 26], [], [7, 15, 13, 16], [], [14, 20, 19, 35], [14, 15], [], [14], [14, 2, 39, 6], [35], [], [], [19, 38, 7], [13, 16, 1, 6], [12, 22, 15, 24], [], [10, 35, 38], "*", [35, 27, 2, 37], [], [10, 18, 32, 7], [7, 18, 25], [], [32], [], [21, 22, 35, 2], [21, 35, 2, 22], [], [35, 32, 1], [2, 19], [], [7, 23], [35, 3, 15, 23], [2], [28, 10, 29, 35]],
    [[5, 35, 3, 31], [5, 8, 13, 30], [4, 29, 23, 10], [10, 28, 24, 35], [10, 35, 2, 39], [10, 14, 18, 39], [36, 39, 34, 10], [10, 39, 35, 34], [10, 13, 28, 38], [8, 35, 40, 5], [10, 36, 3, 37], [35, 29, 3, 5], [21, 4, 30, 40], [35, 28, 31, 40], [28, 27, 3, 18], [27, 16, 18, 38], [21, 36, 29, 31], [], [35, 24, 18, 5], [28, 27, 18, 31], [28, 27, 18, 38], [35, 27, 2, 31], "*", [], [15, 18, 35, 10], [6, 3, 10, 24], [10, 29, 39, 35], [16, 34, 31, 28], [35, 10, 24, 31], [33, 22, 30, 40], [10, 1, 34, 29], [15, 34, 33], [32, 28, 2, 24], [2, 35, 34, 27], [], [35, 10, 28, 24], [35, 18, 10, 13], [35, 10, 18], [28, 35, 10, 23]],
    [[10, 24, 35], [10, 15, 35], [1, 24], [24, 26], [30, 26], [30, 16], [2, 22], [], [13, 26], [], [], [], [], [], [10], [10], [], [], [], [], [10, 19], [], [], "*", [24, 26, 28, 32], [24, 28, 35], [10, 28, 23], [], [], [1], [10, 21, 22], [32], [27, 22], [], [], [], [35, 33], [35], [13, 23, 15]],
    [[10, 35, 20, 28], [10, 20, 35, 26], [], [30, 24, 14, 5], [], [10, 35, 17, 4], [2, 5, 34, 10], [35, 16, 32, 18], [], [10, 37, 36, 5], [37, 36, 4], [14, 10, 34, 17], [35, 3, 22, 5], [29, 3, 28, 18], [10, 28, 18], [28, 20, 10, 16], [35, 28, 21, 18], [19, 1, 26, 17], [35, 38, 19, 18], [1], [35, 20, 10, 6], [10, 5, 18, 32], [15, 18, 35, 10], [24, 26, 28, 32], "*", [35, 38, 18, 16], [10, 30, 4], [24, 34, 28, 32], [24, 26, 28, 18], [35, 18, 34], [35, 22, 18, 39], [35, 28, 34, 4], [4, 28, 10, 34], [32, 1, 10], [35, 28], [6, 29], [18, 28, 32, 10], [24, 28, 35, 30], []],
    [[3, 26, 18, 31], [19, 6, 18, 26], [29, 35], [], [29, 30, 6, 13], [2, 18, 40, 4], [29, 30, 7], [], [10, 19, 29, 38], [14, 29, 18, 36], [10, 14, 36], [35, 14], [15, 32, 35], [3, 27, 16], [3, 35, 10, 40], [3, 35, 31], [3, 17, 30, 39], [], [34, 23, 16, 18], [3, 35, 31], [4, 34, 19], [7, 18, 25], [6, 3, 10, 24], [24, 28, 35], [35, 38, 18, 16], "*", [18, 3, 28, 40], [28, 13], [33, 30], [35, 33, 29, 31], [3, 35, 40, 39], [29, 1, 35, 27], [35, 29, 25, 10], [23, 2, 10, 25], [3, 27], [3, 13, 27, 10], [3, 27, 29, 18], [8, 35], [13, 29, 3, 27]],
    [[1, 3, 11, 27], [10, 28, 8, 3], [10, 14, 29, 40], [15, 29, 28], [], [32, 35, 40, 4], [14, 1, 40, 11], [2, 35, 16], [11, 35, 27, 28], [3, 35, 13, 21], [10, 13, 19, 35], [10, 40, 16], [], [3, 27, 16], [3], [34, 27, 6, 40], [19, 35, 3, 10], [], [19, 21, 11, 27], [10, 36, 23], [19, 24, 26, 31], [], [10, 29, 39, 35], [10, 28, 23], [10, 30, 4], [21, 28, 40, 3], "*", [32, 3, 11, 23], [11, 32, 1], [27, 35, 2, 40], [35, 2, 40, 26], [], [27, 17, 40], [], [13, 35, 8, 24], [13, 35, 1], [27, 40, 28], [11, 13, 27], [13, 35, 29, 38]],
    [[28, 27, 35, 26], [18, 26, 28], [28, 32, 4], [32, 28, 3], [26, 28, 32, 3], [26, 28, 32, 3], [25, 26, 28], [], [28, 32, 1, 24], [35, 10, 23, 24], [6, 28, 25], [28, 32, 1], [32, 35, 13], [], [], [10, 26, 24], [32, 19, 24], [], [], [], [32, 15, 2], [26, 32, 27], [16, 34, 31, 28], [], [24, 34, 28, 32], [28, 13], [5, 11, 1, 23], "*", [], [28, 24, 22, 26], [3, 33, 39, 10], [6, 35, 25, 18], [1, 13, 17, 34], [13, 2, 13, 11], [13, 35, 2], [27, 35, 10, 34], [26, 24, 32, 28], [28, 2, 10, 34], [10, 34, 28, 32]],
    [[28, 35, 26, 18], [28, 35, 27, 9], [10, 28, 29, 37], [23, 2, 10], [28, 33, 29, 32], [2, 29, 18, 36], [32, 23, 2], [35, 25], [10, 28, 32], [28, 19, 34, 36], [3, 35], [32, 30, 40], [30, 18], [3, 27], [3, 27, 40], [], [19, 26], [3, 32], [32, 2], [], [32, 2], [13, 32, 2], [35, 31, 10, 24], [], [32, 26, 28, 18], [32, 30], [11, 32, 1], [], "*", [26, 28, 10, 36], [4, 17, 34, 26], [], [13, 2, 35, 23], [], [], [28], [], [26, 28, 18, 23], [10, 18, 32, 39]],
    [[22, 21, 18, 27], [10, 1, 35, 17], [1, 15, 17, 24], [1, 18], [22, 1, 33, 28], [27, 2, 39, 35], [22, 23, 37, 35], [34, 39, 19, 27], [1, 28, 35, 23], [1, 35, 39, 18], [], [22, 1, 3, 35], [35, 24, 30, 18], [18, 35, 37, 1], [22, 15, 33, 28], [17, 1, 40, 33], [22, 33, 35, 2], [15, 19], [1, 24, 6, 27], [10, 2, 22, 37], [19, 22, 31, 2], [21, 22, 35, 2], [33, 22, 19, 40], [1, 22], [35, 18, 34], [35, 33, 29, 31], [27, 24, 2, 40], [28, 33, 23, 26], [26, 28, 10, 18], "*", [], [24, 35, 2], [2, 25, 28, 39], [35, 10, 2], [35, 11, 22, 31], [22, 19, 29, 40], [22, 19, 29, 40], [33, 3, 34], [22, 35, 13, 24]],
    [[22, 35, 31, 39], [2, 19, 22, 37], [17, 15], [], [17, 2, 18, 39], [], [17, 2, 40], [30, 18, 35, 4], [2, 24, 35, 21], [13, 3, 36, 24], [23, 32, 27, 18], [35, 1], [35, 40, 27, 39], [15, 35, 22, 2], [15, 22, 33, 31], [21, 39, 16, 22], [22, 35, 2, 24], [19, 35, 28, 26], [2, 35, 6], [19, 22, 18], [2, 35, 18], [21, 35, 2, 22], [10], [10, 21, 22], [35, 22, 18, 39], [3, 35, 40, 39], [35, 2, 40, 26], [3, 33, 26], [4, 17, 34, 26], [], "*", [24, 35, 2], [], [], [], [19, 31], [2, 21, 27, 1], [2], [22, 35, 18, 39]],
    [[27, 28, 1, 36], [35, 22, 1, 39], [1, 29, 17], [], [13, 1, 26, 24], [40, 16], [19, 40], [35], [35, 13, 8, 1], [15, 37, 18, 1], [1, 35, 16], [13, 2, 17, 28], [32, 35, 30], [11, 3, 10, 32], [], [35, 10], [26, 27], [28, 26, 19], [28, 26, 27, 1], [], [26, 35, 10], [19, 35], [15, 34, 33], [32], [35, 28, 34, 4], [35, 23, 1, 24], [], [1, 35, 12, 18], [], [2], [2, 5, 13, 16], "*", [2, 5, 13, 16], [35, 1, 11, 9], [2, 13, 15], [27, 26, 1], [6, 28, 11, 1], [8, 28, 1], [35, 1, 10, 28]],
    [[35, 3, 2, 24], [6, 13, 1, 32], [14, 15, 1, 16], [], [15, 17, 13, 16], [], [15, 13, 30, 12], [], [32, 28, 13, 12], [1, 28, 3, 25], [11], [32, 15, 26], [2, 35, 10, 16], [32, 40, 25, 2], [1, 22, 27], [1], [26, 27], [15, 17, 13, 16], [19, 35], [], [35, 2, 10, 34], [2, 19], [32, 28, 2, 24], [4, 28, 10, 34], [4, 28, 10, 34], [1, 35, 27], [17, 27, 8, 40], [25, 13, 2, 34], [13, 2, 35, 23], [2, 25, 28, 39], [], [2, 5, 13, 16], "*", [12, 26, 1, 32], [15, 34, 1, 16], [32, 26, 12, 17], [], [1, 34, 12, 3], []],
    [[2, 27, 28, 11], [2, 27, 28, 11], [1, 28, 10], [3, 18, 31], [15, 13, 10, 1], [15, 16], [25, 2, 35, 11], [1], [34, 9], [11, 15], [13], [1, 13, 2, 4], [2, 35], [11, 1, 2, 9], [11, 29, 28, 27], [1], [], [13], [15, 1, 28, 16], [], [15, 10, 32, 2], [15, 1, 32, 19], [2, 35, 34, 27], [], [32, 1, 10, 25], [2, 28, 10, 25], [11, 10, 1, 16], [13, 2, 10], [], [35, 10, 2, 16], [], [1, 35, 11, 10], [1, 12, 26, 15], "*", [7, 1, 4, 16], [35, 1, 13, 11], [], [34, 35, 7, 13], [13, 2, 10]],
    [[29, 5, 15, 8], [19, 15, 29], [35, 1, 29, 2], [1, 35], [15, 30], [15, 16], [15, 29], [], [35, 10, 14], [15, 17, 20], [35, 16], [15, 37, 1, 8], [35, 30, 14], [35, 3, 32, 6], [35, 13], [2, 16], [2, 27, 3, 35], [6, 22, 26, 1], [19, 35, 29, 13], [], [19, 34], [18, 15, 1], [15, 10, 2, 13], [], [35, 28], [3, 35, 15], [35, 13, 8, 24], [35, 5, 1, 10], [], [35, 11, 32, 31], [], [1, 13, 31], [15, 34, 1, 16], [1, 16, 7, 4], "*", [15, 29, 37, 28], [1], [27, 34, 35], [35, 28, 6, 37]],
    [[26, 30, 36, 34], [1, 10, 26, 39], [1, 19, 26, 24], [1, 26], [13], [6, 36], [34, 26, 6], [1, 16], [34, 10, 28], [26, 16], [], [29, 13, 28, 15], [2, 22, 17, 19], [2, 13, 28], [10, 4, 28, 15], [], [2, 17, 13], [24, 17, 13], [27, 2, 29, 28], [], [19, 30, 34], [10, 35, 13, 2], [35, 10, 28, 29], [], [6, 29], [13, 3, 27, 10], [13, 35, 1], [2, 26, 10, 34], [26, 24, 32], [22, 19, 29, 40], [], [27, 26, 1, 13], [27, 9, 26, 24], [1, 13], [29, 15, 28, 37], "*", [15, 10, 37, 28], [], [12, 17, 28]],
    [[28, 29, 26, 32], [25, 28, 17, 15], [17, 24, 26, 16], [], [2, 36, 26, 18], [2, 35, 30, 18], [29, 1, 4, 16], [2, 18, 26, 31], [3, 4, 16, 35], [30, 28, 40, 19], [2, 36, 37], [27, 13, 1, 39], [11, 22, 39, 30], [27, 3, 15, 28], [19, 29, 39, 25], [25, 34, 6, 35], [3, 27, 35, 16], [2, 24, 26], [35, 38], [19, 35, 16, 25], [19, 35, 16], [35, 3, 15, 19], [1, 18, 10, 24], [35, 33, 27, 22], [18, 28, 32, 9], [3, 27, 29, 18], [27, 40, 28, 8], [26, 24, 32, 28], [], [22, 19, 29, 28], [2, 21], [5, 28, 11, 29], [], [12, 26], [15, 10, 37, 28], [], "*", [34, 2, 1], [35, 18]],
    [[26, 35, 18, 19], [2, 26, 35], [14, 4, 28, 29], [], [14, 30, 28, 23], [23], [35, 34, 16, 24], [], [10, 18], [2, 35], [13, 35], [19, 32, 1, 13], [], [25, 13], [], [], [26, 17, 19], [8, 32, 19], [23, 2, 13], [], [27, 28], [2, 3, 28], [35, 10, 18, 5], [35, 33], [24, 28, 35, 30], [35, 13], [11, 27, 32], [28, 2, 10, 34], [26, 28, 18, 23], [2, 33], [2], [12, 6, 13], [1, 12, 34, 3], [13, 5], [27, 4, 1, 35], [15, 24, 10], [34, 27, 25], "*", [5, 12, 35, 26]],
    [[35, 3, 24, 37], [1, 28, 15, 35], [18, 4, 28, 38], [30, 7, 14, 26], [10, 26, 34, 2], [10, 15, 17, 7], [10, 6, 2, 34], [35, 37, 10, 2], [], [28, 15, 10, 36], [10, 37, 14], [14, 10, 34, 40], [35, 3, 22, 39], [29, 28, 10, 18], [35, 10, 2, 18], [20, 10, 16, 38], [15, 28, 35], [2, 25, 16], [35, 10, 38, 19], [1], [28, 35, 10], [28, 10, 29, 35], [28, 10, 35, 23], [13, 15, 23], [], [8, 35], [13, 35, 10, 38], [1, 10, 34, 28], [18, 10, 32, 1], [22, 35, 13, 24], [35, 22, 18, 39], [35, 28, 2, 24], [1, 28, 7, 10], [13, 2, 10, 25], [13, 5, 28, 37], [12, 17, 28, 24], [35, 18, 27, 2], [5, 12, 35, 26], "*"]
]


# Load and embed data
@st.cache_resource
def load_data(csv_path):
    logger.debug(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        logger.debug("Data loaded successfully.")
        return df, index, model
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None, None, None

# Extract problematics
def extract_problematics(text):
    logger.debug("Extracting problematics...")
    text = text.lower()
    keywords = set()
    tech_terms = ["processor", "memory", "cache", "speed", "execution", "pipeline", "latency", "delay", "power", "energy", 
                  "efficiency", "performance", "throughput", "temperature", "heat", "reliability", "stability", "error", 
                  "instruction", "queue", "system", "hardware", "network", "interface", "data", "processing", "feedback", 
                  "ai", "dataset", "compatibility"]
    for term in tech_terms:
        if term in text:
            keywords.add(term)
    
    prompt = (
        f"Given the following text from patent documents:\n\n{text[:2000]}\n\n"
        f"Extracted keywords: {', '.join(keywords)}\n\n"
        f"Analyze the text and keywords to identify 3-5 precise problematics—specific contradictions or challenges "
        f"in the described systems or methods. Each problematic must: (1) Include a quantifiable improvement metric "
        f"(e.g., speed, latency, throughput) directly supported by the text, (2) Pair it with a clear constraint or trade-off "
        f"explicitly or implicitly present in the text (e.g., without increasing resources, without reducing flexibility), "
        f"and (3) Be distinct, avoiding overlap. Do not invent percentages or specific numbers unless explicitly stated "
        f"in the text. List only the problematics, one per line:"
    )
    response = generate_response(prompt)
    
    problematics = [line.strip() for line in response.split('\n') if line.strip() and not line.startswith(('Debug:', 'Based on', 'Here', 'Inference'))]
    if not problematics or len(problematics) < 2:
        problematics = ["Increase processing speed without exhausting memory capacity", 
                        "Reduce execution latency without increasing system complexity"]
    logger.debug(f"Extracted problematics: {problematics}")
    return problematics

# LLM-driven TRIZ mapping
def map_to_triz(problematic, text):
    logger.debug(f"Mapping to TRIZ for problematic: {problematic[:100]}...")
    prompt = (
        f"Given the following problematic derived from patent text:\n\n{problematic}\n\n"
        f"Context from patent text:\n\n{text[:2000]}\n\n"
        f"Below is a list of TRIZ parameters (1-39):\n\n{', '.join([f'{k}: {v}' for k, v in TRIZ_PARAMS.items()])}\n\n"
        f"Analyze the problematic and text to identify: (1) the 'Improving' TRIZ parameter (what’s being enhanced), "
        f"and (2) the 'Worsening' TRIZ parameter (the trade-off or constraint). Select parameters directly supported by "
        f"the text and problematic’s intent. Provide only the parameter numbers (e.g., '39, 26') on a single line, "
        f"without explanation or additional text:"
    )
    response = generate_response(prompt)
    
    try:
        improving, worsening = map(int, response.strip().split(','))
        if improving not in TRIZ_PARAMS or worsening not in TRIZ_PARAMS:
            raise ValueError
    except (ValueError, IndexError):
        improving, worsening = 39, 26  # Defaults: Productivity, Quantity of substance
    logger.debug(f"TRIZ parameters: Improving={improving}, Worsening={worsening}")
    return improving, worsening

# Dual solutions
def get_solutions(problematic, chunks, improving, worsening):
    logger.debug(f"Generating solutions for problematic: {problematic[:100]}...")
    principles_idx = TRIZ_MATRIX[improving-1][worsening-1] if improving and worsening and TRIZ_MATRIX[improving-1][worsening-1] else [1]
    principle_names = [principles[i-1] for i in principles_idx]
    
    strict_prompt = (
        f"Problematic: {problematic}\n"
        f"Patent Insights (for context, do not copy): {' '.join(chunks[:2])}\n"
        f"TRIZ Principles (use only these, select at least one): {', '.join(principle_names)}\n"
        f"Suggest a novel, concise solution that resolves the problematic using only the listed TRIZ principles. "
        f"Focus on reducing costs, complexity, and size while ensuring reliability. Do not replicate existing patent solutions; "
        f"create a new approach inspired by but distinct from the patent insights. Limit to 100 words:"
    )
    strict_solution = generate_response(strict_prompt)
    
    inspired_prompt = (
        f"Problematic: {problematic}\n"
        f"Patent Insights (use these as the basis): {' '.join(chunks[:2])}\n"
        f"Analyze the patent text to identify a compact technology innovation. Propose a concise enhancement for heat management, "
        f"addressing cost, complexity, and reliability. Limit to 100 words and build on the patent’s innovation:"
    )
    inspired_solution = generate_response(inspired_prompt)
    
    if not isinstance(strict_solution, str) or not strict_solution.strip():
        strict_solution = "Default TRIZ solution: Apply segmentation to optimize resource use."
    if not isinstance(inspired_solution, str) or not inspired_solution.strip():
        inspired_solution = "Default Patents solution: Enhance existing method with predictive caching."
    
    logger.debug(f"Solutions generated - TRIZ: {strict_solution[:50]}..., Patents: {inspired_solution[:50]}...")
    return strict_solution, inspired_solution, principle_names

# Export results to CSV (single problematic)
def export_results(problematic, improving, worsening, principles, strict_solution, inspired_solution):
    logger.debug("Exporting single problematic results...")
    try:
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Problematic", "Improving Parameter", "Worsening Parameter", "TRIZ Principles", "Strict Solution", "Inspired Solution"])
        writer.writerow([problematic, TRIZ_PARAMS.get(improving, 'N/A'), TRIZ_PARAMS.get(worsening, 'N/A'), ", ".join(principles), strict_solution, inspired_solution])
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error in export_results: {e}")
        st.error(f"Error in export_results: {e}")
        return ""

# Evaluation Functions
def evaluate_clarity(text):
    try:
        if not isinstance(text, str) or not text.strip():
            logger.warning("Clarity evaluation: Invalid text input")
            return 0.0
        blob = TextBlob(text)
        words = len(word_tokenize(text))
        sentences = len(blob.sentences)
        syllables = sum(sum(1 for char in word.lower() if char in 'aeiouy') for word in word_tokenize(text))
        if sentences == 0 or words == 0:
            logger.warning("Clarity evaluation: Zero sentences or words")
            return 0.0
        flesch = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        flesch = max(0.0, min(100.0, flesch))
        word_count = len(word_tokenize(text))
        conciseness = 100.0 if word_count <= 30 else max(50.0, 100.0 - (word_count - 30) * 2)
        score = 0.5 * flesch + 0.5 * conciseness
        logger.debug(f"Clarity score: {score}")
        return float(score)
    except Exception as e:
        logger.error(f"Error in evaluate_clarity: {e}")
        return 0.0

def evaluate_relevance(problematic, output, context, model):
    try:
        if not all(isinstance(x, str) and x.strip() for x in [problematic, output, context]):
            logger.warning("Relevance evaluation: Invalid inputs")
            return 0.0
        embeddings = model.encode([problematic, output, context], batch_size=8)
        prob_emb, out_emb, ctx_emb = embeddings
        prob_sim = cosine_similarity([prob_emb], [out_emb])[0][0]
        ctx_sim = cosine_similarity([out_emb], [ctx_emb])[0][0]
        score = (0.6 * prob_sim + 0.4 * ctx_sim) * 100
        logger.debug(f"Relevance score: {score}")
        return float(score)
    except Exception as e:
        logger.error(f"Error in evaluate_relevance: {e}")
        return 0.0

def evaluate_novelty(output, context, model):
    try:
        if not all(isinstance(x, str) and x.strip() for x in [output, context]):
            logger.warning("Novelty evaluation: Invalid inputs")
            return 0.0
        embeddings = model.encode([output, context], batch_size=8)
        out_emb, ctx_emb = embeddings
        sim = cosine_similarity([out_emb], [ctx_emb])[0][0]
        novelty = (1 - sim) * 100
        word_count = len(word_tokenize(output))
        penalty = 1.0 if word_count >= 10 else 0.5
        score = novelty * penalty
        logger.debug(f"Novelty score: {score}")
        return float(score)
    except Exception as e:
        logger.error(f"Error in evaluate_novelty: {e}")
        return 0.0

def evaluate_feasibility(solution, problematic):
    try:
        if not all(isinstance(x, str) and x.strip() for x in [solution, problematic]) or nlp is None:
            logger.warning("Feasibility evaluation: Invalid inputs or spaCy model")
            return 0.0
        doc = nlp(solution)
        complex_words = len([token for token in doc if len(token.text) > 8])
        complexity = complex_words / max(len(word_tokenize(solution)), 1)
        feasibility = 100.0 - (complexity * 50)
        keywords = set(word_tokenize(problematic.lower())) & set(["power", "cost", "resource", "complexity"])
        if keywords:
            feasibility *= 0.8
        score = max(0.0, min(100.0, feasibility))
        logger.debug(f"Feasibility score: {score}")
        return float(score)
    except Exception as e:
        logger.error(f"Error in evaluate_feasibility: {e}")
        return 0.0

def evaluate_solution(problematic, strict_solution, inspired_solution, context, model):
    logger.debug(f"Evaluating solutions for problematic: {problematic[:100]}...")
    try:
        results = {
            "TRIZ Clarity": evaluate_clarity(strict_solution),
            "TRIZ Relevance": evaluate_relevance(problematic, strict_solution, context, model),
            "TRIZ Novelty": evaluate_novelty(strict_solution, context, model),
            "TRIZ Feasibility": evaluate_feasibility(strict_solution, problematic),
            "Patents Clarity": evaluate_clarity(inspired_solution),
            "Patents Relevance": evaluate_relevance(problematic, inspired_solution, context, model),
            "Patents Novelty": evaluate_novelty(inspired_solution, context, model),
            "Patents Feasibility": evaluate_feasibility(inspired_solution, problematic),
            "TRIZ Total": 0.0,
            "Patents Total": 0.0,
            "Overall Score": 0.0,
            "Total": 0.0
        }
        # Calculate TRIZ and Patents totals
        triz_total = np.mean([results["TRIZ Clarity"], results["TRIZ Relevance"], results["TRIZ Novelty"], results["TRIZ Feasibility"]])
        patents_total = np.mean([results["Patents Clarity"], results["Patents Relevance"], results["Patents Novelty"], results["Patents Feasibility"]])
        # Calculate overall score for the double solution
        overall_score = np.mean([triz_total, patents_total])
        # Existing Total (average of all metrics)
        total = np.mean([v for k, v in results.items() if k not in ["TRIZ Total", "Patents Total", "Overall Score", "Total"]])
        results["TRIZ Total"] = float(triz_total)
        results["Patents Total"] = float(patents_total)
        results["Overall Score"] = float(overall_score)
        results["Total"] = float(total)
        logger.debug(f"Evaluation results: {results}")
        return results
    except Exception as e:
        logger.error(f"Error evaluating solutions: {e}")
        st.error(f"Error evaluating solutions: {e}")
        return {
            "TRIZ Clarity": 0.0, "TRIZ Relevance": 0.0, "TRIZ Novelty": 0.0, "TRIZ Feasibility": 0.0,
            "Patents Clarity": 0.0, "Patents Relevance": 0.0, "Patents Novelty": 0.0, "Patents Feasibility": 0.0,
            "TRIZ Total": 0.0, "Patents Total": 0.0, "Overall Score": 0.0, "Total": 0.0
        }

# Save to Solutions CSV
def save_to_solutions_csv(problematic, strict_solution, inspired_solution):
    logger.debug("Saving to solutions CSV...")
    if not all(isinstance(x, str) and x.strip() for x in [problematic, strict_solution, inspired_solution]):
        logger.error("Invalid or empty solutions. Skipping save.")
        st.error("Error: Invalid or empty solutions. Skipping save.")
        return False
    file_exists = os.path.isfile("triz_solutions.csv")
    try:
        with open("triz_solutions.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Problematic", "Solution 1 (TRIZ)", "Solution 2 (Patents)"])
            writer.writerow([problematic, strict_solution, inspired_solution])
        logger.debug("Saved to triz_solutions.csv")
        return True
    except Exception as e:
        logger.error(f"Error saving to solutions CSV: {e}")
        st.error(f"Error saving to solutions CSV: {e}")
        return False

# Save to Evaluation CSV
def save_to_evaluation_csv(problematic, results):
    logger.debug("Saving to evaluation CSV...")
    if not isinstance(problematic, str) or not problematic.strip() or not results:
        logger.error("Invalid problematic or evaluation results. Skipping save.")
        st.error("Error: Invalid problematic or evaluation results. Skipping save.")
        return False
    file_exists = os.path.isfile("triz_evaluation.csv")
    try:
        with open("triz_evaluation.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Problematic", "TRIZ Clarity", "TRIZ Relevance", "TRIZ Novelty", "TRIZ Feasibility", 
                                 "Patents Clarity", "Patents Relevance", "Patents Novelty", "Patents Feasibility", 
                                 "TRIZ Total", "Patents Total", "Overall Score", "Total"])
            writer.writerow([problematic, 
                             results["TRIZ Clarity"], results["TRIZ Relevance"], results["TRIZ Novelty"], results["TRIZ Feasibility"], 
                             results["Patents Clarity"], results["Patents Relevance"], results["Patents Novelty"], results["Patents Feasibility"], 
                             results["TRIZ Total"], results["Patents Total"], results["Overall Score"], results["Total"]])
        logger.debug("Saved to triz_evaluation.csv")
        return True
    except Exception as e:
        logger.error(f"Error saving to evaluation CSV: {e}")
        st.error(f"Error saving to evaluation CSV: {e}")
        return False

# Export Solutions to Excel
def export_solutions_to_excel():
    logger.debug("Exporting solutions to Excel...")
    try:
        if os.path.isfile("triz_solutions.csv"):
            df_solutions = pd.read_csv("triz_solutions.csv", encoding="utf-8")
            df_solutions.to_excel("triz_solutions.xlsx", index=False)
            logger.debug("Exported triz_solutions.xlsx")
            return True
        else:
            st.error("No solutions found to export.")
            return False
    except Exception as e:
        logger.error(f"Error exporting solutions to Excel: {e}")
        st.error(f"Error exporting to Excel: {e}")
        return False

# Export Evaluations to Excel
def export_evaluations_to_excel():
    logger.debug("Exporting evaluations to Excel...")
    try:
        if os.path.isfile("triz_evaluation.csv"):
            df_eval = pd.read_csv("triz_evaluation.csv", encoding="utf-8")
            df_eval.to_excel("triz_evaluation.xlsx", index=False)
            logger.debug("Exported triz_evaluation.xlsx")
            return True
        else:
            st.error("No evaluations found to export.")
            return False
    except Exception as e:
        logger.error(f"Error exporting evaluations to Excel: {e}")
        st.error(f"Error exporting to Excel: {e}")
        return False

# Export Short Evaluation to Excel
def export_short_evaluation_to_excel():
    logger.debug("Exporting short evaluation to Excel...")
    try:
        if not os.path.isfile("triz_solutions.csv"):
            st.error("No solutions found to export.")
            return False
        
        # Load solutions
        df_solutions = pd.read_csv("triz_solutions.csv", encoding="utf-8")
        
        # Initialize Total Score column
        df_solutions["Total Score"] = "Not Evaluated"
        
        # If evaluations exist, merge Total scores
        if os.path.isfile("triz_evaluation.csv"):
            df_eval = pd.read_csv("triz_evaluation.csv", encoding="utf-8")
            # Merge on Problematic
            df_merged = df_solutions.merge(
                df_eval[["Problematic", "Total"]],
                on="Problematic",
                how="left"
            )
            # Update Total Score where available
            df_solutions["Total Score"] = df_merged["Total"].apply(
                lambda x: f"{x:.1f}" if pd.notnull(x) else "Not Evaluated"
            )
        
        # Save to Excel
        df_solutions.to_excel("short_evaluation.xlsx", index=False)
        logger.debug("Exported short_evaluation.xlsx")
        return True
    except Exception as e:
        logger.error(f"Error exporting short evaluation to Excel: {e}")
        st.error(f"Error exporting to Excel: {e}")
        return False

# Streamlit UI
def main():
    st.title("TRIZ Patent Innovation Engine")
    st.markdown("Analyze patent texts to identify contradictions and propose innovative solutions using TRIZ methodology.")

    # Initialize session state
    if 'solutions_generated' not in st.session_state:
        st.session_state.solutions_generated = False
    if 'problematic' not in st.session_state:
        st.session_state.problematic = None
    if 'strict_solution' not in st.session_state:
        st.session_state.strict_solution = None
    if 'inspired_solution' not in st.session_state:
        st.session_state.inspired_solution = None
    if 'principle_names' not in st.session_state:
        st.session_state.principle_names = None
    if 'improving' not in st.session_state:
        st.session_state.improving = None
    if 'worsening' not in st.session_state:
        st.session_state.worsening = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'text' not in st.session_state:
        st.session_state.text = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # Input Section
    with st.container():
        st.header("Step 1: Input Patent Query")
        st.markdown("Enter a specific problematic or leave blank to analyze a random patent contradiction.")
        query = st.text_input(
            "Query (e.g., 'Improve cache hit rate without increasing power consumption')",
            placeholder="Type your query here or leave blank for random analysis",
            help="Enter a technical contradiction or optimization goal."
        )
        analyze_button = st.button("Generate Solutions")

    # Generate Solutions
    if analyze_button:
        logger.debug("Starting analysis...")
        with st.spinner("Loading patent data..."):
            df, index, model = load_data("all_patents_corpus.csv")
            if df is None:
                st.error("Failed to load patent data. Please check the CSV file.")
                return
            st.session_state.model = model

        with st.spinner("Extracting problematic..."):
            if query:
                problematic = query
                chunks = [text for text in df["text"] if any(word in text.lower() for word in query.lower().split())][:2]
                text = " ".join(chunks)
            else:
                sample_texts = df["text"].sample(5).tolist()
                problematics = extract_problematics("\n".join(sample_texts))
                problematic = random.choice(problematics)
                chunks = sample_texts
                text = " ".join(sample_texts)

        # TRIZ Parameters
        improving, worsening = map_to_triz(problematic, text)
        principles_idx = TRIZ_MATRIX[improving-1][worsening-1] if improving and worsening and TRIZ_MATRIX[improving-1][worsening-1] else [1]
        principle_names = [principles[i-1] for i in principles_idx]

        # Generate Solutions
        with st.spinner("Generating solutions with Nemotron 70B..."):
            strict_solution, inspired_solution, principle_names = get_solutions(problematic, chunks, improving, worsening)

        # Save to session state
        st.session_state.solutions_generated = True
        st.session_state.problematic = problematic
        st.session_state.strict_solution = strict_solution
        st.session_state.inspired_solution = inspired_solution
        st.session_state.principle_names = principle_names
        st.session_state.improving = improving
        st.session_state.worsening = worsening
        st.session_state.chunks = chunks
        st.session_state.text = text
        st.session_state.results = None

        # Save to solutions CSV
        save_to_solutions_csv(problematic, strict_solution, inspired_solution)
        st.success("Solutions generated and saved!")

    # Display Generated Solutions
    if st.session_state.solutions_generated:
        with st.container():
            st.header("Step 2: Review Generated Solutions")
            
            # Problematic
            with st.expander("Problematic", expanded=True):
                st.markdown(f"**Identified Contradiction**: {st.session_state.problematic}")

            # TRIZ Parameters
            with st.expander("TRIZ Parameters"):
                st.markdown(f"**Improving Parameter**: {TRIZ_PARAMS.get(st.session_state.improving, 'N/A')} (#{st.session_state.improving})")
                st.markdown(f"**Worsening Parameter**: {TRIZ_PARAMS.get(st.session_state.worsening, 'N/A')} (#{st.session_state.worsening})")

            # TRIZ Principles
            with st.expander("TRIZ Principles (Strict Mode)"):
                st.markdown(f"**Principles**: {', '.join(st.session_state.principle_names)}")

            # Proposed Solutions
            with st.expander("Proposed Solutions", expanded=True):
                st.subheader("Solution 1 (TRIZ)")
                st.markdown(st.session_state.strict_solution)
                st.subheader("Solution 2 (Patents)")
                st.markdown(st.session_state.inspired_solution)

    # Export Solutions
    with st.container():
        st.header("Step 3: Export Solutions")
        if st.button("Export Solutions to Excel"):
            if export_solutions_to_excel():
                with open("triz_solutions.xlsx", "rb") as f:
                    st.download_button(
                        label="Download Solutions Excel",
                        data=f,
                        file_name="triz_solutions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error("Failed to export solutions.")

    # Evaluate Solutions
    with st.container():
        st.header("Step 4: Evaluate Solutions")
        if st.session_state.solutions_generated:
            if st.button("Evaluate Solutions"):
                with st.spinner("Evaluating solutions..."):
                    results = evaluate_solution(
                        st.session_state.problematic,
                        st.session_state.strict_solution,
                        st.session_state.inspired_solution,
                        st.session_state.text,
                        st.session_state.model
                    )
                    st.session_state.results = results
                    st.success("Evaluation completed! Review below.")
        else:
            st.warning("Please generate solutions first.")

    # Display Evaluation Results
    if st.session_state.results:
        with st.container():
            st.header("Step 5: Review Evaluation Results")
            with st.expander("Evaluation Results", expanded=True):
                st.markdown(f"### Evaluation for: {st.session_state.problematic[:100]}...")
                st.markdown("**Metrics Explanation**:")
                st.markdown("- **Clarity**: How readable and concise the solution is (Flesch score + word count).")
                st.markdown("- **Relevance**: Alignment with the problematic and patent context (cosine similarity).")
                st.markdown("- **Novelty**: Uniqueness compared to patent context (cosine distance).")
                st.markdown("- **Feasibility**: Practicality, penalized for complexity or resource demands.")
                st.markdown("- **TRIZ Total**: Average of TRIZ metrics (0-100).")
                st.markdown("- **Patents Total**: Average of Patents metrics (0-100).")
                st.markdown("- **Overall Score**: Average of TRIZ Total and Patents Total (0-100).")
                st.markdown("- **Total**: Average of all metrics (0-100).")

                # Radar Plot
                st.subheader("Criteria Comparison (Radar Plot)")
                categories = ['Clarity', 'Relevance', 'Novelty', 'Feasibility']
                triz_data = [
                    st.session_state.results["TRIZ Clarity"],
                    st.session_state.results["TRIZ Relevance"],
                    st.session_state.results["TRIZ Novelty"],
                    st.session_state.results["TRIZ Feasibility"]
                ]
                patents_data = [
                    st.session_state.results["Patents Clarity"],
                    st.session_state.results["Patents Relevance"],
                    st.session_state.results["Patents Novelty"],
                    st.session_state.results["Patents Feasibility"]
                ]
                # Repeat the first value to close the radar chart
                triz_data += [triz_data[0]]
                patents_data += [patents_data[0]]
                categories += [categories[0]]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=triz_data,
                    theta=categories,
                    fill='toself',
                    name='TRIZ Solution',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatterpolar(
                    r=patents_data,
                    theta=categories,
                    fill='toself',
                    name='Patents Solution',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Evaluation Criteria Comparison"
                )
                st.plotly_chart(fig)

                st.subheader("TRIZ Solution Metrics")
                triz_metrics = [
                    ("Clarity", st.session_state.results["TRIZ Clarity"], "Readability and conciseness of TRIZ solution."),
                    ("Relevance", st.session_state.results["TRIZ Relevance"], "Alignment with problematic and context."),
                    ("Novelty", st.session_state.results["TRIZ Novelty"], "Uniqueness of TRIZ solution."),
                    ("Feasibility", st.session_state.results["TRIZ Feasibility"], "Practicality of TRIZ solution.")
                ]
                triz_df = pd.DataFrame(triz_metrics, columns=["Metric", "Score", "Description"])
                st.table(triz_df)

                st.subheader("Patents Solution Metrics")
                patents_metrics = [
                    ("Clarity", st.session_state.results["Patents Clarity"], "Readability and conciseness of Patents solution."),
                    ("Relevance", st.session_state.results["Patents Relevance"], "Alignment with problematic and context."),
                    ("Novelty", st.session_state.results["Patents Novelty"], "Uniqueness of Patents solution."),
                    ("Feasibility", st.session_state.results["Patents Feasibility"], "Practicality of Patents solution.")
                ]
                patents_df = pd.DataFrame(patents_metrics, columns=["Metric", "Score", "Description"])
                st.table(patents_df)

                st.subheader("Overall")
                st.markdown(f"**TRIZ Total**: {st.session_state.results['TRIZ Total']:.1f}/100")
                st.markdown(f"**Patents Total**: {st.session_state.results['Patents Total']:.1f}/100")
                st.markdown(f"**Overall Score (Double Solution)**: {st.session_state.results['Overall Score']:.1f}/100")
                st.markdown(f"**Total Score**: {st.session_state.results['Total']:.1f}/100")
                for metric, score in st.session_state.results.items():
                    if metric not in ["TRIZ Total", "Patents Total", "Overall Score", "Total"]:
                        status = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
                        progress_value = min(float(score) / 100.0, 1.0)
                        logger.debug(f"Progress value for {metric}: {progress_value}, type: {type(progress_value)}")
                        st.markdown(f"- **{metric}**: {score:.1f} ({status})")
                        st.progress(progress_value, text=status)
                
                if st.session_state.results["Total"] >= 80:
                    st.success("High-quality output!")
                elif st.session_state.results["Total"] >= 60:
                    st.warning("Good, but consider refining novelty or clarity.")
                else:
                    st.error("Needs improvement in solution quality.")

    # Save Evaluation
    with st.container():
        st.header("Step 6: Save Evaluation")
        if st.session_state.results:
            if st.button("Save Evaluation"):
                if save_to_evaluation_csv(st.session_state.problematic, st.session_state.results):
                    if export_evaluations_to_excel():
                        with open("triz_evaluation.xlsx", "rb") as f:
                            st.download_button(
                                label="Download Evaluation Excel",
                                data=f,
                                file_name="triz_evaluation.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        st.success("Evaluation saved and exported!")
                    else:
                        st.error("Evaluation saved but failed to export to Excel.")
                else:
                    st.error("Failed to save evaluation.")
        else:
            st.warning("Please evaluate solutions first.")

    # Save Short Evaluation
    with st.container():
        st.header("Step 7: Save Short Evaluation")
        st.markdown("Export a summary of all problematics, their solutions, and Total scores to Excel.")
        if st.button("Save Short Evaluation as Excel"):
            if export_short_evaluation_to_excel():
                with open("short_evaluation.xlsx", "rb") as f:
                    st.download_button(
                        label="Download Short Evaluation Excel",
                        data=f,
                        file_name="short_evaluation.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                st.success("Short evaluation exported successfully!")
            else:
                st.error("Failed to export short evaluation.")

    # Stored Solutions Section
    if os.path.isfile("triz_solutions.csv"):
        with st.container():
            st.header("Stored Solutions")
            try:
                df = pd.read_csv("triz_solutions.csv", encoding="utf-8")
                st.dataframe(df[["Problematic", "Solution 1 (TRIZ)", "Solution 2 (Patents)"]], use_container_width=True)
            except Exception as e:
                logger.error(f"Error reading triz_solutions.csv: {e}")
                st.error(f"Error reading triz_solutions.csv: {e}")

    # Evaluation Summary
    if os.path.isfile("triz_evaluation.csv"):
        with st.container():
            st.header("Evaluation Summary")
            try:
                df_eval = pd.read_csv("triz_evaluation.csv", encoding="utf-8")
                avg_total = df_eval["Total"].mean()
                precision_score = (df_eval["TRIZ Relevance"].mean() * 0.5 + df_eval["Patents Relevance"].mean() * 0.5)
                
                st.markdown("### General Evaluation")
                st.markdown(f"- **Average Total Score Across All Problematics**: {avg_total:.1f}/100")
                st.markdown(f"- **Average Precision (Solution Relevance)**: {precision_score:.1f}/100")
                for metric in df_eval.columns:
                    if metric not in ["Problematic", "TRIZ Total", "Patents Total", "Overall Score", "Total"]:
                        avg_score = df_eval[metric].mean()
                        st.markdown(f"- **Average {metric}**: {avg_score:.1f}")
                
                if avg_total >= 80:
                    st.success("Model outputs are high-quality overall!")
                elif avg_total >= 60:
                    st.warning("Good performance. Refine `get_solutions` prompts for higher novelty or relevance.")
                else:
                    st.error("Model needs improvement. Adjust `map_to_triz` prompts or FAISS embeddings.")
            except Exception as e:
                logger.error(f"Error reading triz_evaluation.csv: {e}")
                st.error(f"Error reading triz_evaluation.csv: {e}")

    # Export Single Result as CSV
    if st.session_state.solutions_generated:
        with st.container():
            st.header("Export Single Analysis as CSV")
            csv_output = export_results(
                st.session_state.problematic,
                st.session_state.improving,
                st.session_state.worsening,
                st.session_state.principle_names,
                st.session_state.strict_solution,
                st.session_state.inspired_solution
            )
            if csv_output:
                st.download_button(
                    label="Download Analysis as CSV",
                    data=csv_output,
                    file_name="triz_analysis.csv",
                    mime="text/csv",
                    help="Download the problematic, parameters, principles, and solutions as a CSV file."
                )

    # Clear Stored Data
    with st.container():
        st.header("Clear Data")
        if st.button("Clear Stored Data"):
            for file in ["triz_solutions.csv", "triz_evaluation.csv", "triz_solutions.xlsx", "triz_evaluation.xlsx"]:
                if os.path.isfile(file):
                    os.remove(file)
            # Reset session state
            st.session_state.solutions_generated = False
            st.session_state.problematic = None
            st.session_state.strict_solution = None
            st.session_state.inspired_solution = None
            st.session_state.principle_names = None
            st.session_state.improving = None
            st.session_state.worsening = None
            st.session_state.chunks = None
            st.session_state.text = None
            st.session_state.results = None
            st.session_state.model = None
            st.rerun()

if __name__ == "__main__":
    main()
