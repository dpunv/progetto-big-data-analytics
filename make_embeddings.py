#!/usr/bin/env python3
"""
Text embedding script using Hugging Face transformers with MPS acceleration.
Reads lines from input file, generates embeddings, and saves to output file.
Supports checkpointing every 100k embeddings and resuming from checkpoints.
"""

import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings from token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_existing_embeddings(output_file: str):
    """Load existing embeddings from file if it exists."""
    if Path(output_file).exists():
        print(f"Found existing embeddings file: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} existing embeddings")
        return data
    return []


def save_embeddings(output_file: str, results: list):
    """Save embeddings to file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def load_checkpoint(checkpoint_file: str):
    """Load checkpoint file if it exists."""
    if Path(checkpoint_file).exists():
        print(f"Found checkpoint file: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded checkpoint with {len(data)} embeddings")
        return data
    return []


def save_checkpoint(checkpoint_file: str, results: list):
    """Save checkpoint file."""
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Checkpoint saved: {len(results)} embeddings")


def generate_embeddings(input_file: str, output_file: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings for each line in the input file.
    
    Args:
        input_file: Path to input text file
        output_file: Path to output JSON file
        model_name: Name of the embedding model to use
    """
    # Setup checkpoint file
    checkpoint_file = str(Path(output_file).with_suffix('')) + '_checkpoint.json'
    checkpoint_interval = 100000
    
    # Load existing embeddings or checkpoint
    results = load_checkpoint(checkpoint_file)
    if not results:
        results = load_existing_embeddings(output_file)
    
    # Create a set of already processed line numbers for fast lookup
    processed_lines = {item['line_number'] for item in results}
    starting_count = len(results)
    
    if processed_lines:
        print(f"Resuming from line {max(processed_lines) + 1}")
    
    # Check for MPS (Metal) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Metal (MPS) GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (no GPU acceleration)")
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f]
    
    # Filter out already processed lines
    lines_to_process = []
    line_numbers = []
    for idx, line in enumerate(all_lines, 1):
        if line.strip() and idx not in processed_lines:
            lines_to_process.append(line.strip())
            line_numbers.append(idx)
    
    total_lines = len(all_lines)
    new_lines = len(lines_to_process)
    
    print(f"Total lines in file: {total_lines}")
    print(f"Already processed: {len(processed_lines)}")
    print(f"Lines to process: {new_lines}")
    
    if new_lines == 0:
        print("✓ All lines already processed!")
        return
    
    # Process in batches
    batch_size = 32
    embeddings_since_checkpoint = 0
    
    with torch.no_grad():
        for i in range(0, new_lines, batch_size):
            batch = lines_to_process[i:i+batch_size]
            batch_line_numbers = line_numbers[i:i+batch_size]
            
            # Tokenize
            encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # Generate embeddings
            model_output = model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Move to CPU and convert to list
            embeddings = embeddings.cpu().numpy()
            
            for line_num, line, embedding in zip(batch_line_numbers, batch, embeddings):
                results.append({
                    "line_number": line_num,
                    "text": line,
                    "embedding": embedding.tolist()
                })
                embeddings_since_checkpoint += 1
            
            # Save checkpoint every 100k embeddings
            if embeddings_since_checkpoint >= checkpoint_interval:
                save_checkpoint(checkpoint_file, results)
                embeddings_since_checkpoint = 0
            
            # Progress update
            processed_so_far = min(i + batch_size, new_lines)
            total_processed = starting_count + processed_so_far
            if processed_so_far % 1000 < batch_size or processed_so_far >= new_lines:
                print(f"Processed {processed_so_far}/{new_lines} new lines (total: {total_processed}/{total_lines})")
    
    # Sort results by line number before final save
    results.sort(key=lambda x: x['line_number'])
    
    # Final save
    print(f"Saving final results to: {output_file}")
    save_embeddings(output_file, results)
    
    # Remove checkpoint file after successful completion
    if Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()
        print("✓ Checkpoint file removed")
    
    print(f"✓ Done! Generated {len(results)} total embeddings")
    print(f"  New embeddings: {new_lines}")
    print(f"  Embedding dimension: {len(results[0]['embedding'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for text lines using transformers on Apple Silicon"
    )
    parser.add_argument(
        "input_file",
        help="Input text file (one line per entry)"
    )
    parser.add_argument(
        "-o", "--output",
        default="embeddings.json",
        help="Output JSON file (default: embeddings.json)"
    )
    parser.add_argument(
        "-m", "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    generate_embeddings(args.input_file, args.output, args.model)
    return 0


if __name__ == "__main__":
    exit(main())