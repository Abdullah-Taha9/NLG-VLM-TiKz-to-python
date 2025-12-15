"""Load and filter the DaTikZ dataset."""

from datasets import load_dataset
from config import DATA_LIMIT


def load_data(limit: int = DATA_LIMIT):
    """
    Load DaTikZ dataset, filter arxiv origin, and limit rows.
    
    Args:
        limit: Maximum number of rows to return
        
    Returns:
        Filtered dataset with arxiv samples only
    """
    # Load dataset
    dataset = load_dataset("nllg/datikz-v3", split="train")
    
    # Filter arxiv origin only
    dataset = dataset.filter(lambda x: x["origin"] == "arxiv")
    
    # Limit rows
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    return dataset
