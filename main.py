"""Main entry point for TikZ to Matplotlib pipeline."""

from data_loader import load_data
from pipeline import process_dataset, save_results


def main():
    """Run the full pipeline."""
    print("Loading dataset...")
    dataset = load_data()
    print(f"Loaded {len(dataset)} arxiv samples")
    
    print("Processing with Gemini...")
    results = process_dataset(dataset)
    
    print("Saving results...")
    output_path = save_results(results)
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    main()
