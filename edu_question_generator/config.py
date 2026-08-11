"""Configuration for Edu Question Generator: DeepSeek API, document chunking, and question-count targets."""
import os

# --- DeepSeek API connection ---
# Actual model is read from .streamlit/secrets.toml (DEEPSEEK_MODEL) via ui.get_deepseek_model()
# Common models: deepseek-chat | deepseek-reasoner | deepseek-v4-pro | deepseek-v4-flash
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))

# --- Fixed character-based chunking (internal fallback in chunking for very long segments) ---
CHUNK_SIZE = 650
CHUNK_OVERLAP = 100

# --- Exam pipeline targets (large documents) ---
TARGET_QUESTIONS_TOTAL = int(os.getenv("TARGET_QUESTIONS_TOTAL", "20"))
TARGET_COMPUTATION_MIN = int(os.getenv("TARGET_COMPUTATION_MIN", "6"))
TARGET_ANALYSIS_APPLICATION_MIN = int(os.getenv("TARGET_ANALYSIS_APPLICATION_MIN", "8"))
MAX_LOGICAL_SEGMENTS = int(os.getenv("MAX_LOGICAL_SEGMENTS", "10"))
LOGICAL_SEGMENT_MAX_CHARS = int(os.getenv("LOGICAL_SEGMENT_MAX_CHARS", "14000"))

# Unified error codes raised by llm_client / pipeline
LLM_REQUEST_TOO_LARGE = "LLM_REQUEST_TOO_LARGE"
LLM_LIMIT_ERROR = "LLM_LIMIT_ERROR"
LLM_INSUFFICIENT_BALANCE = "LLM_INSUFFICIENT_BALANCE"
PIPELINE_ALL_SEGMENTS_FAILED = "PIPELINE_ALL_SEGMENTS_FAILED"
LLM_INVALID_MODEL = "LLM_INVALID_MODEL"  # Unsupported model name (HTTP 400)
