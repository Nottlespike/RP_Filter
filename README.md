# RP_Filter

A quality filtering tool for roleplay dialogue datasets using LLM-based evaluation.

## Description

RP_Filter processes JSONL files containing roleplay conversations and uses an LLM to evaluate their quality based on writing metrics. It filters the dataset to keep only the highest quality conversations.

## Usage

```bash
python dataset_filter.py input_file.jsonl output_file.jsonl [options]
```

### Required Arguments
- `input_file`: Path to input JSONL file containing conversations
- `output_file`: Path where filtered JSONL output will be saved

### Optional Arguments
```bash
--api-base        LLM API base URL (default: http://localhost:8000/v1)
--batch-size      Number of conversations to process per batch (default: 4)
--top-percentage  Percentage of highest quality conversations to keep (default: 0.05)
--max-parallel    Maximum number of parallel API requests (default: 8)
--model          Name of the LLM model to use (default: local-model)
--verbose        Enable detailed logging
```

## Input Format

The input JSONL file should contain conversations in ShareGPT JSONL:


## Quality Evaluation Criteria

The LLM evaluates each conversation on:

1. Vocabulary Diversity (1-10)
   - Word choice variety
   - Language sophistication
   - Repetition avoidance

2. Purple Prose Quality (1-10)
   - Description richness
   - Emotional depth
   - Imagery and metaphors

3. Context Length Usage (1-10)
   - Dialogue development
   - Scenario complexity
   - Conversation flow

## Example

```bash
# Basic usage
python dataset_filter.py conversations.jsonl filtered_output.jsonl

# With custom options
python dataset_filter.py \
    conversations.jsonl \
    filtered_output.jsonl \
    --api-base "http://localhost:5000/v1" \
    --batch-size 8 \
    --top-percentage 0.1 \
    --max-parallel 16 \
    --model "gpt-4" \
    --verbose
```

## Requirements

- Python 3.7+

## Error Handling

- Failed API calls automatically retry with exponential backoff
- Invalid JSON lines are skipped and logged
- Progress bar shows batch processing status
- Detailed error logging with --verbose flag