"""Gradio app for live TikZ to Matplotlib generation using LLM pipeline."""

import sys
import os

# Add parent directory to path to import chain module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset

from chain import invoke_chain
from config import DATA_LIMIT


def load_arxiv_data(limit: int = DATA_LIMIT):
    """Load DaTikZ dataset, filter arxiv origin, and limit rows."""
    dataset = load_dataset("nllg/datikz-v3", split="train")
    dataset = dataset.filter(lambda x: x["origin"] == "arxiv")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def execute_code(python_code: str) -> Image.Image:
    """Execute Python code and capture the matplotlib figure as an image."""
    plt.close('all')
    
    namespace = {"plt": plt, "__builtins__": __builtins__}
    
    try:
        exec(python_code, namespace)
        fig = plt.gcf()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        
        img = Image.open(buf).copy()
        buf.close()
        plt.close('all')
        
        return img
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Error:\n{str(e)[:100]}", 
                ha='center', va='center', fontsize=10, color='red',
                transform=ax.transAxes, wrap=True)
        ax.axis('off')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white')
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        plt.close('all')
        
        return img


# Load dataset once
print("Loading dataset...")
DATA = load_arxiv_data()
TOTAL = len(DATA)
print(f"Loaded {TOTAL} samples")


def get_sample_info(index: int):
    """Get sample info without generating."""
    sample = DATA[index]
    original_img = sample.get("image")
    caption = sample.get("caption", "No caption")
    tikz_code = sample.get("code", "")
    
    return original_img, caption, tikz_code, f"Sample {index + 1} / {TOTAL}"


def generate_matplotlib(index: int):
    """Run LLM pipeline and generate matplotlib image."""
    sample = DATA[index]
    
    # Call the LLM chain
    python_code = invoke_chain(
        image=sample["image"],
        caption=sample["caption"],
        tikz_code=sample["code"]
    )
    
    # Execute the generated code
    generated_img = execute_code(python_code)
    
    return generated_img, python_code


def navigate(current_idx: int, direction: int):
    """Navigate to previous or next sample."""
    new_idx = (current_idx + direction) % TOTAL
    orig, caption, tikz, counter = get_sample_info(new_idx)
    return new_idx, orig, caption, tikz, counter, None, ""


# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# TikZ → Matplotlib Live Generator")
    gr.Markdown("*Select a sample and click Generate to run the LLM pipeline*")
    
    # State for current index
    current_index = gr.State(value=0)
    
    # Counter
    counter = gr.Markdown(f"**Sample 1 / {TOTAL}**")
    
    # Caption and TikZ code
    with gr.Row():
        caption_text = gr.Textbox(label="Caption", lines=2, interactive=False)
        tikz_code = gr.Textbox(label="TikZ Code", lines=5, interactive=False)
    
    # Images: Original on top, Generated below
    with gr.Column():
        gr.Markdown("### Original (TikZ)")
        original_image = gr.Image(label="Original", height=300)
        
        generate_btn = gr.Button("🚀 Generate Matplotlib Code", variant="primary", size="lg")
        
        gr.Markdown("### Generated (Matplotlib)")
        generated_image = gr.Image(label="Generated", height=300)
        
        generated_code = gr.Code(label="Generated Python Code", language="python")
    
    # Navigation buttons
    with gr.Row():
        prev_btn = gr.Button("◀ Previous")
        next_btn = gr.Button("Next ▶")
    
    # Button handlers
    def go_prev(idx):
        return navigate(idx, -1)
    
    def go_next(idx):
        return navigate(idx, 1)
    
    def run_generate(idx):
        gen_img, code = generate_matplotlib(idx)
        return gen_img, code
    
    prev_btn.click(
        fn=go_prev,
        inputs=[current_index],
        outputs=[current_index, original_image, caption_text, tikz_code, counter, generated_image, generated_code]
    )
    
    next_btn.click(
        fn=go_next,
        inputs=[current_index],
        outputs=[current_index, original_image, caption_text, tikz_code, counter, generated_image, generated_code]
    )
    
    generate_btn.click(
        fn=run_generate,
        inputs=[current_index],
        outputs=[generated_image, generated_code]
    )
    
    # Load first sample on startup
    def init():
        orig, caption, tikz, cnt = get_sample_info(0)
        return orig, caption, tikz, cnt
    
    app.load(
        fn=init,
        outputs=[original_image, caption_text, tikz_code, counter]
    )


if __name__ == "__main__":
    app.launch()
