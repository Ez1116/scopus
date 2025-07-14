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
import time
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
DEFAULT_BATCH_SIZE = 50
MAX_RETRIES = 3

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

async def evaluate_single_batch(articles_batch: list, client: genai.Client, config: dict, batch_num: int, total_batches: int, max_retries: int = MAX_RETRIES):
    """Evaluate a single batch of articles with retry logic."""
    batch_size = len(articles_batch)
    
    # Create the batch input content
    articles_json = json.dumps(articles_batch, ensure_ascii=False, indent=2)
    
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

Evaluate ALL {batch_size} articles and assign a relevance score from 0-5 based on how well each article relates to the research topic. Return ONLY the JSON array, no other text."""

    from google.genai import types
    
    for attempt in range(max_retries):
        try:
            print(f"[INFO] Processing batch {batch_num}/{total_batches} (articles {articles_batch[0]['article_id']}-{articles_batch[-1]['article_id']}, attempt {attempt + 1}/{max_retries})")
            
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
                        finish_reason = candidate.finish_reason
                        if finish_reason == "MAX_TOKENS":
                            print(f"[WARN] Batch {batch_num} hit MAX_TOKENS limit - may need smaller batches")
                        else:
                            print(f"[INFO] Batch {batch_num} finish reason: {finish_reason}")
                        
                    if hasattr(candidate, 'content') and candidate.content:
                        parts = candidate.content.parts
                        
                        if parts:
                            for part in parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                        else:
                            print(f"[WARN] Batch {batch_num} response has no content parts")
                    else:
                        print(f"[WARN] Batch {batch_num} candidate {i} has no content")
            else:
                print(f"[WARN] Batch {batch_num} has no candidates in response")
                
            if response_text:
                print(f"[INFO] Batch {batch_num} received response text ({len(response_text)} characters)")
            
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
                
                # Extract scores and create result mapping
                batch_scores = {}
                
                for evaluation in evaluations:
                    if isinstance(evaluation, dict):
                        article_id = evaluation.get("article_id", 0)
                        relevance_score = evaluation.get("relevance_score", 0)
                        batch_scores[article_id] = int(relevance_score)
                
                print(f"✅ Batch {batch_num} completed: {len(evaluations)} evaluations parsed")
                return batch_scores
                
            except json.JSONDecodeError as json_err:
                print(f"[ERROR] Batch {batch_num} JSON parse failed (attempt {attempt + 1}): {json_err}")
                if response_text:
                    print(f"[ERROR] Response preview: {response_text[:200]}...")
                
                # If this is the last attempt, return zeros for this batch
                if attempt == max_retries - 1:
                    print(f"[ERROR] Batch {batch_num} failed after {max_retries} attempts - using default scores")
                    return {article['article_id']: 0 for article in articles_batch}
                
                # Wait before retry
                await asyncio.sleep(1)
                
        except Exception as exc:
            print(f"[ERROR] Batch {batch_num} API call failed (attempt {attempt + 1}): {exc}")
            
            # If this is the last attempt, return zeros for this batch
            if attempt == max_retries - 1:
                print(f"[ERROR] Batch {batch_num} failed after {max_retries} attempts - using default scores")
                return {article['article_id']: 0 for article in articles_batch}
            
            # Wait before retry
            await asyncio.sleep(2)
    
    # This should never be reached, but just in case
    return {article['article_id']: 0 for article in articles_batch}

def get_progress_file_path(output_path: str) -> str:
    """Generate progress file path based on output file path."""
    base_name = os.path.splitext(output_path)[0]
    return f"{base_name}_progress.json"

def save_progress(progress_file: str, completed_batches: dict, total_batches: int):
    """Save current progress to file."""
    progress_data = {
        "completed_batches": completed_batches,
        "total_batches": total_batches,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

def load_progress(progress_file: str) -> dict:
    """Load progress from file if it exists."""
    if not os.path.exists(progress_file):
        return {}
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

async def batch_evaluate_all_articles(df: pd.DataFrame, client: genai.Client, config: dict, title_col: str, abstract_col: str, output_path: str, batch_size: int = DEFAULT_BATCH_SIZE, start_time: float = None):
    """Process all articles in intelligent batches with progress tracking and error recovery."""
    
    total_articles = len(df)
    progress_file = get_progress_file_path(output_path)
    
    print(f"Processing {total_articles} articles in batches of {batch_size}...")
    
    # Prepare all articles as structured data
    all_articles = []
    for idx, row in df.iterrows():
        all_articles.append({
            "article_id": idx + 1,
            "title": row[title_col],
            "abstract": row[abstract_col]
        })
    
    # Split into batches
    batches = []
    for i in range(0, len(all_articles), batch_size):
        batch = all_articles[i:i + batch_size]
        batches.append(batch)
    
    total_batches = len(batches)
    print(f"Split into {total_batches} batches")
    
    # Load existing progress
    progress_data = load_progress(progress_file)
    all_scores = {}
    completed_batch_count = 0
    
    if progress_data.get("completed_batches"):
        print(f"Found existing progress file - resuming from previous session")
        # Convert string keys back to integers
        for article_id_str, score in progress_data["completed_batches"].items():
            all_scores[int(article_id_str)] = score
        completed_batch_count = len([batch for batch in batches if all(article['article_id'] in all_scores for article in batch)])
        if completed_batch_count > 0:
            print(f"Resuming: {completed_batch_count} batches already completed")
    
    # Process batches sequentially with progress tracking
    for batch_idx, batch in enumerate(batches, 1):
        # Check if this batch is already completed
        batch_already_done = all(article['article_id'] in all_scores for article in batch)
        
        if batch_already_done:
            print(f"[SKIP] Batch {batch_idx}/{total_batches} already completed")
        else:
            batch_scores = await evaluate_single_batch(
                batch, client, config, batch_idx, total_batches
            )
            all_scores.update(batch_scores)
            
            # Save progress after each batch
            save_progress(progress_file, all_scores, total_batches)
        
        # Show progress with elapsed time and estimated remaining time
        processed_articles = batch_idx * batch_size
        if processed_articles > total_articles:
            processed_articles = total_articles
        
        progress_percent = processed_articles / total_articles * 100
        
        # Calculate elapsed time and estimated remaining time
        if start_time:
            elapsed_time = time.time() - start_time
            elapsed_minutes = elapsed_time / 60
            
            if processed_articles > 0:
                time_per_article = elapsed_time / processed_articles
                remaining_articles = total_articles - processed_articles
                estimated_remaining_time = time_per_article * remaining_articles
                estimated_remaining_minutes = estimated_remaining_time / 60
                
                print(f"Progress: {processed_articles}/{total_articles} articles ({progress_percent:.1f}%) | "
                      f"Elapsed: {elapsed_minutes:.1f}m | Est. remaining: {estimated_remaining_minutes:.1f}m")
            else:
                print(f"Progress: {processed_articles}/{total_articles} articles ({progress_percent:.1f}%) | "
                      f"Elapsed: {elapsed_minutes:.1f}m")
        else:
            print(f"Progress: {processed_articles}/{total_articles} articles ({progress_percent:.1f}%)")
    
    # Convert to ordered list matching original dataframe
    final_scores = []
    for idx in range(len(df)):
        article_id = idx + 1
        score = all_scores.get(article_id, 0)
        final_scores.append(score)
    
    # Clean up progress file on successful completion
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Removed progress file: {progress_file}")
    
    print(f"✅ Batch evaluation completed! Processed {len(final_scores)} articles across {total_batches} batches.")
    return final_scores

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
    
    # Track start time for timeout and elapsed time display
    start_time = time.time()
    print(f"Starting processing with {args.timeout} second timeout ({args.timeout/60:.1f} minutes)")
    
    # Process articles in intelligent batches with timeout
    try:
        scores = await asyncio.wait_for(
            batch_evaluate_all_articles(df, client, config, title_col, abstract_col, args.output, start_time=start_time),
            timeout=args.timeout
        )
        elapsed_time = time.time() - start_time
        print(f"✅ Processing completed successfully in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        print(f"\n⏰ TIMEOUT: Processing stopped after {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"Progress has been saved and processing can be resumed by running the same command again.")
        
        # Attempt to load any partial results from progress file for graceful exit
        progress_file = get_progress_file_path(args.output)
        progress_data = load_progress(progress_file)
        
        if progress_data.get("completed_batches"):
            completed_articles = len(progress_data["completed_batches"])
            total_articles = len(df)
            print(f"Completed: {completed_articles}/{total_articles} articles ({completed_articles/total_articles*100:.1f}%)")
        
        # Exit gracefully - don't save incomplete results to final output
        print("Use the same command to resume processing from where it left off.")
        return
    

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
    parser.add_argument('--timeout', type=int, default=600, help='Maximum execution time in seconds (default: 600 = 10 minutes)')
    parsed_args = parser.parse_args()

    try:
        asyncio.run(main(parsed_args))
    except KeyboardInterrupt:
        print("Interrupted by user.")