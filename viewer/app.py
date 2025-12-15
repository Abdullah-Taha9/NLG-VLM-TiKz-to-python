"""Gradio app for comparing original TikZ images with generated Matplotlib images."""

import json
import io
import gradio as gr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# Path to dataset
DATA_PATH = "../output/dataset.json"


def load_data():
    """Load processed dataset from JSON."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_code(python_code: str) -> Image.Image:
    """
    Execute Python code and capture the matplotlib figure as an image.
    
    Args:
        python_code: Python/Matplotlib code to execute
        
    Returns:
        PIL Image of the generated figure
    """
    # Clear any existing figures
    plt.close('all')
    
    # Create a namespace for execution
    namespace = {"plt": plt, "__builtins__": __builtins__}
    
    try:
        # Execute the code
        exec(python_code, namespace)
        
        # Get current figure
        fig = plt.gcf()
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Convert to PIL Image
        img = Image.open(buf).copy()
        buf.close()
        plt.close('all')
        
        return img
    except Exception as e:
        # Return error image
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


def get_sample(index: int, data: list):
    """Get a sample by index and return both images."""
    sample = data[index]
    
    # Original image - load from saved PNG file
    image_path = sample.get("image_path")
    if image_path:
        full_path = f"../output/{image_path}"
        try:
            original_img = Image.open(full_path)
        except Exception:
            original_img = None
    else:
        original_img = None
    
    # Generated image - execute the python code
    generated_img = execute_code(sample.get("python_code", ""))
    
    caption = sample.get("caption", "No caption")
    
    return original_img, generated_img, caption, f"Sample {index + 1} / {len(data)}"


# Load data once
DATA = load_data()
TOTAL = len(DATA)


def navigate(current_idx: int, direction: int):
    """Navigate to previous or next sample."""
    new_idx = (current_idx + direction) % TOTAL
    orig, gen, caption, counter = get_sample(new_idx, DATA)
    return new_idx, orig, gen, caption, counter


# Custom CSS for minimal white design
# CSS = """
# .gradio-container {
#     background-color: white !important;
#     max-width: 1200px !important;
#     margin: auto !important;
# }
# .image-container {
#     background-color: white !important;
#     border: 1px solid #e0e0e0 !important;
#     border-radius: 8px !important;
# }
# .nav-button {
#     font-size: 24px !important;
#     min-width: 60px !important;
# }
# """

# Build Gradio interface
# with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as app:
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# TikZ → Matplotlib Comparison")
    
    # State for current index
    current_index = gr.State(value=0)
    
    # Counter
    counter = gr.Markdown(f"**Sample 1 / {TOTAL}**")
    
    # Caption
    caption_text = gr.Markdown("*Loading...*")
    
    # Images side by side
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original (TikZ)")
            original_image = gr.Image(label="Original", height=400)
        with gr.Column():
            gr.Markdown("### Generated (Matplotlib)")
            generated_image = gr.Image(label="Generated", height=400)
    
    # Navigation buttons
    with gr.Row():
        prev_btn = gr.Button("◀ Previous", elem_classes=["nav-button"])
        next_btn = gr.Button("Next ▶", elem_classes=["nav-button"])
    
    # Button handlers
    def go_prev(idx):
        return navigate(idx, -1)
    
    def go_next(idx):
        return navigate(idx, 1)
    
    prev_btn.click(
        fn=go_prev,
        inputs=[current_index],
        outputs=[current_index, original_image, generated_image, caption_text, counter]
    )
    
    next_btn.click(
        fn=go_next,
        inputs=[current_index],
        outputs=[current_index, original_image, generated_image, caption_text, counter]
    )
    
    # Load first sample on startup
    def init():
        orig, gen, caption, cnt = get_sample(0, DATA)
        return orig, gen, caption, cnt
    
    app.load(
        fn=init,
        outputs=[original_image, generated_image, caption_text, counter]
    )


if __name__ == "__main__":
    app.launch()
