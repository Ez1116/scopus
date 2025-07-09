""" scopus-abstract.py

Flexible Academic Abstract Relevance Evaluator
-----------------------------------------------
• Reads a CSV file that must include at least two columns:
    - 'title'      : Article title
    - 'abstract'   : Abstract text

• Uses Google Gemini `2.0-flash` model to score each article on a 0‑5 scale
  based on configurable research criteria defined in JSON configuration files.

• Scoring criteria are fully customizable through configuration files:
    - Default: GenAI-era assessment countermeasures (config_genai_assessment.json)
    - Custom: Create your own research focus using the template (config_template.json)

• Asynchronously streams up to 50 requests in parallel for speed.

• Saves results to a new CSV with columns:
    - title
    - relevance_score  (0‑5 integer)
    - DOI             (if available in input)
    - Link            (if available in input)

Prerequisites
--------------
pip install --upgrade google-genai pandas python-dotenv tqdm

Set the environment variable `GEMINI_API_KEY` *or* create a .env file
with `GEMINI_API_KEY=your-gemini-api-key`.

Usage
-----
# Default GenAI assessment configuration
python scopus-abstract.py \
       --input  scopus_AI.csv \
       --output scopus_AI_evaluations.csv

# Using a custom configuration
python scopus-abstract.py \
       --input  my_data.csv \
       --output my_results.csv \
       --config my_custom_config.json

# Creating a custom configuration
cp config_template.json my_research_config.json
# Edit my_research_config.json to match your research focus
python scopus-abstract.py \
       --input  data.csv \
       --output results.csv \
       --config my_research_config.json
"""

import argparse
import asyncio
import csv
import json
import os
# from typing import List  # No longer needed
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

import pandas as pd
from google import genai
load_dotenv()

# ---------------------------- configuration ---------------------------- #

MODEL_NAME = "gemini-2.0-flash"
CONCURRENT_REQUESTS = 50
DEFAULT_TEMPERATURE = 0
MAX_TOKENS = 8192
DEFAULT_CONFIG_FILE = "config_genai_assessment.json"

# ---------------------------- configuration loading ----------------------------- #

def load_config(config_file=None):
    """Load configuration from JSON file."""
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def validate_config(config):
    """Validate configuration structure."""
    required_fields = [
        'research_topic', 'tool_name', 'tool_description', 
        'scoring_criteria', 'system_prompt_template', 'evaluation_instruction'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in configuration: {field}")
    
    # Validate scoring criteria has all required scores
    required_scores = ['0', '1', '2', '3', '4', '5']
    for score in required_scores:
        if score not in config['scoring_criteria']:
            raise ValueError(f"Missing score '{score}' in scoring_criteria")
    
    return True

def create_batch_evaluation_function(config):
    """Create batch evaluation function from configuration.""" 
    
    def batch_evaluate_articles(article_1_id: int, article_1_title: str, article_1_score: int,
                               article_2_id: int, article_2_title: str, article_2_score: int,
                               article_3_id: int, article_3_title: str, article_3_score: int,
                               article_4_id: int, article_4_title: str, article_4_score: int,
                               article_5_id: int, article_5_title: str, article_5_score: int) -> str:
        """Record relevance scores for articles (simplified version with first 5 articles).
        
        Args:
            article_1_id: ID of first article
            article_1_title: Title of first article  
            article_1_score: Relevance score (0-5) for first article
            article_2_id: ID of second article
            article_2_title: Title of second article
            article_2_score: Relevance score (0-5) for second article
            article_3_id: ID of third article
            article_3_title: Title of third article
            article_3_score: Relevance score (0-5) for third article
            article_4_id: ID of fourth article
            article_4_title: Title of fourth article
            article_4_score: Relevance score (0-5) for fourth article
            article_5_id: ID of fifth article
            article_5_title: Title of fifth article
            article_5_score: Relevance score (0-5) for fifth article
        
        Returns:
            Confirmation message
        """
        return f"Evaluated 5 articles successfully"
    
    return batch_evaluate_articles

# ---------------------- helper: build message list --------------------- #

# The make_messages function is no longer needed as we handle content directly in evaluate_row

# ----------------------------- main logic ----------------------------- #

async def batch_evaluate_all_articles(df: pd.DataFrame, client: genai.Client, config: dict, title_col: str, abstract_col: str):
    """Send one batch request to Gemini and return scores for all articles."""
    
    # Prepare all articles as JSON data
    articles_data = []
    for idx, row in df.iterrows():
        articles_data.append({
            "article_id": idx + 1,
            "title": row[title_col],
            "abstract": row[abstract_col]
        })
    
    # Create the batch input content
    articles_json = json.dumps(articles_data, ensure_ascii=False, indent=2)
    
    scoring_description = "\\n".join([
        f"{score}: {desc}" 
        for score, desc in config['scoring_criteria'].items()
    ])
    
    user_content = f"""Please evaluate the relevance of the following articles based on the research topic: {config['research_topic']}

Research Topic: {config['research_topic']}

Scoring Criteria:
{scoring_description}

Articles to evaluate:
{articles_json}

Please evaluate each article and return your response as a JSON array with the following format:
[
  {{"article_id": 1, "title": "Article title", "relevance_score": 3}},
  {{"article_id": 2, "title": "Article title", "relevance_score": 1}},
  ...
]

Evaluate ALL {len(articles_data)} articles and assign a relevance score from 0-5 based on how well each article relates to the research topic. Return ONLY the JSON array, no other text."""

    from google.genai import types
    
    try:
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": user_content}]}],
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=0
            )
        )

        # Extract the text response
        response_text = ""
        
        if hasattr(response, 'candidates') and response.candidates:
            for i, candidate in enumerate(response.candidates):
                # Check for finish reason
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    print(f"[INFO] Response finish reason: {candidate.finish_reason}")
                    
                if hasattr(candidate, 'content') and candidate.content:
                    parts = candidate.content.parts
                    
                    if parts:
                        for part in parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                    else:
                        print("[WARN] Response has no content parts")
                else:
                    print(f"[WARN] Candidate {i} has no content")
        else:
            print("[WARN] No candidates found in response")
            
        if response_text:
            print(f"[INFO] Received response text ({len(response_text)} characters)")
        
        # Try to parse JSON response
        try:
            # Clean the response text to extract just the JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            evaluations = json.loads(response_text)
            
            # Extract scores in the correct order
            scores = [0] * len(df)  # Initialize with zeros
            
            for evaluation in evaluations:
                if isinstance(evaluation, dict):
                    article_id = evaluation.get("article_id", 0)
                    relevance_score = evaluation.get("relevance_score", 0)
                    
                    # Convert to proper index (article_id is 1-based)
                    if 1 <= article_id <= len(df):
                        scores[article_id - 1] = int(relevance_score)
            
            print(f"✅ Successfully parsed {len(evaluations)} evaluations from response")
            return scores
            
        except json.JSONDecodeError as json_err:
            print(f"[ERROR] Failed to parse JSON response: {json_err}")
            if response_text:
                print(f"[ERROR] Response preview: {response_text[:200]}...")
            return [0] * len(df)
            
    except Exception as exc:
        print(f"[ERROR] Batch evaluation failed: {exc}")
        return [0] * len(df)

async def main(args):
    # Load configuration
    try:
        config = load_config(args.config)
        validate_config(config)
        print(f"Loaded configuration for: {config['research_topic']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        return

    # Load data
    df = pd.read_csv(args.input)
    
    # Check for required columns (support both lowercase and Title case)
    required_columns = []
    if 'title' in df.columns:
        required_columns.append('title')
    elif 'Title' in df.columns:
        required_columns.append('Title')
    else:
        raise ValueError("Input CSV must contain 'title' or 'Title' column.")
    
    if 'abstract' in df.columns:
        required_columns.append('abstract')
    elif 'Abstract' in df.columns:
        required_columns.append('Abstract')
    else:
        raise ValueError("Input CSV must contain 'abstract' or 'Abstract' column.")
    
    # Check for optional DOI and Link columns
    doi_col = None
    if 'DOI' in df.columns:
        doi_col = 'DOI'
    elif 'doi' in df.columns:
        doi_col = 'doi'
    
    link_col = None
    if 'Link' in df.columns:
        link_col = 'Link'
    elif 'link' in df.columns:
        link_col = 'link'
    
    # Store column names for later use
    title_col = required_columns[0]
    abstract_col = required_columns[1]

    client = genai.Client()
    
    # Print processing information
    print(f"Processing {len(df)} articles in a single batch request...")
    
    # Single batch evaluation call
    scores = await batch_evaluate_all_articles(df, client, config, title_col, abstract_col)
    
    print(f"✅ Batch evaluation completed! Processed {len(scores)} articles.")

    # Save to new CSV
    output_data = {
        'title': df[title_col],
        'relevance_score': scores
    }
    
    # Add DOI and Link columns if they exist in the input
    if doi_col:
        output_data['DOI'] = df[doi_col]
    if link_col:
        output_data['Link'] = df[link_col]
    
    out_df = pd.DataFrame(output_data)
    out_df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible academic abstract relevance evaluator.")
    parser.add_argument('--input',  required=True, help='Input CSV (with title, abstract)')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--config', default=DEFAULT_CONFIG_FILE, help=f'Configuration file (default: {DEFAULT_CONFIG_FILE})')
    parsed_args = parser.parse_args()

    try:
        asyncio.run(main(parsed_args))
    except KeyboardInterrupt:
        print("Interrupted by user.")