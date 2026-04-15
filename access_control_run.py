"""
Access Control Policy — DAG-to-Knowledge-Graph Pipeline

Converts Access Control DAG images into structured knowledge graphs
using OpenAI-compatible vision models.

Setup:
    pip install -r requirements.txt
    cp .env.example .env          # then fill in your OPENAI_API_KEY

Usage:
    python access_control_run.py --method extract_entities
    python access_control_run.py --method relation_classification --entities_input <PATH>
    python access_control_run.py --method relation_extraction --input <IMAGE_DIR> --output <OUT_DIR>
    python access_control_run.py --help

See README.md for full documentation.
"""

from src.cli import main

if __name__ == "__main__":
    main()
