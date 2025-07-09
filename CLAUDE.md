# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python script that analyzes academic article abstracts. It uses Google's Gemini API to score abstracts on a 0-5 relevance scale.

## Dependencies and Setup

Install required packages:
```bash
pip install --upgrade google-genai pandas python-dotenv tqdm
```

Environment setup:
- Set `GEMINI_API_KEY` environment variable OR create `.env` file with `GEMINI_API_KEY=your-gemini-api-key`
- The script uses Google's "gemini-2.0-flash" model

## Usage

Run the script with:
```bash
python scopus-abstract.py --input <csv_file> --output <output_file> --config <config_file>
```

**Basic usage with default GenAI assessment configuration:**
```bash
python scopus-abstract.py --input scopus_AI.csv --output scopus_AI_evaluations.csv
```

**Using a custom configuration:**
```bash
python scopus-abstract.py --input my_data.csv --output my_results.csv --config my_custom_config.json
```

**Input CSV requirements:**
- Must contain 'title' and 'abstract' columns

**Output CSV format:**
- `title`: Article title
- `relevance_score`: Integer 0-5 relevance score

## Configuration System

The script now supports flexible configuration through JSON files:

**Default configuration:** `config_genai_assessment.json` - Pre-configured for GenAI assessment research
**Template configuration:** `config_template.json` - Template for creating custom research configurations

**Configuration file structure:**
- `research_topic`: Description of the research focus
- `tool_name`: Name of the evaluation function (usually "record_evaluation")
- `tool_description`: How the evaluation tool works
- `scoring_criteria`: Detailed 0-5 scoring rubric
- `system_prompt_template`: AI model instructions
- `evaluation_instruction`: User instruction for evaluation

## Architecture

**Single-file design** with key components:
- **Batch processing**: Processes all articles in a single API request for efficiency
- **Gemini integration**: Uses JSON response parsing for structured scoring
- **Configuration loading**: JSON-based flexible topic configuration
- **Error handling**: Graceful handling of API response parsing failures
- **Progress tracking**: Simple progress feedback for batch processing

**Key constants:**
- `MODEL_NAME = "gemini-2.0-flash"`
- `CONCURRENT_REQUESTS = 50` (legacy, no longer used)
- `MAX_TOKENS = 8192`

## Customizing for Different Research Topics

To adapt the script for different research topics:

1. **Copy the template:** `cp config_template.json my_research_config.json`
2. **Edit the configuration:** Modify all fields to match your research focus
3. **Run with custom config:** `python scopus-abstract.py --input data.csv --output results.csv --config my_research_config.json`

## Development Notes

- No formal test suite exists
- Dependencies are documented in code comments only
- Configuration is now externalized in JSON files
- Error handling defaults scores to 0 on API failures
- Uses Google Gemini's function calling for structured responses
- Configuration validation ensures all required fields are present