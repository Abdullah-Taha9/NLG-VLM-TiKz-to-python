"""Pipeline to process dataset row-by-row."""

import os
import json
import time
from tqdm import tqdm

from config import DELAY_SECONDS, OUTPUT_DIR
from chain import invoke_chain


def process_row(row: dict) -> dict:
    """
    Process a single row: invoke LLM and add python_code.
    
    Args:
        row: Dataset row with image, caption, code
        
    Returns:
        Row with added python_code field
    """
    python_code = invoke_chain(
        image=row["image"],
        caption=row["caption"],
        tikz_code=row["code"]
    )
    
    return {**row, "python_code": python_code}


def process_dataset(dataset) -> list:
    """
    Process entire dataset row-by-row with rate limiting.
    
    Args:
        dataset: HuggingFace dataset
        
    Returns:
        List of processed rows with python_code added
    """
    results = []
    
    for row in tqdm(dataset, desc="Processing"):
        processed = process_row(row)
        results.append(processed)
        
        # Rate limiting
        time.sleep(DELAY_SECONDS)
    
    return results


def save_results(results: list) -> str:
    """
    Save processed results to JSON file and images to images/ folder.
    
    Args:
        results: List of processed rows
        
    Returns:
        Path to saved JSON file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, "dataset.json")
    
    serializable_results = []
    for i, row in enumerate(results):
        # Save image as PNG file
        image_filename = f"sample_{i:04d}.png"
        image_path = os.path.join(images_dir, image_filename)
        
        if row.get("image"):
            row["image"].save(image_path, "PNG")
        
        # Create serializable row with image path instead of PIL object
        row_copy = {k: v for k, v in row.items() if k != "image"}
        row_copy["image_path"] = f"images/{image_filename}"
        serializable_results.append(row_copy)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    return output_path
