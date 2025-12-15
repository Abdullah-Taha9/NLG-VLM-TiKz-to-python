# Live TikZ to Matplotlib Generator

This is a Gradio app that allows users to generate Matplotlib images from TikZ code using a LLM pipeline.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export GEMINI_API_KEY=your_gemini_api_key
```

3. Run the app:
```bash
python app.py
```

### Make sure Latex engine is installed (some python codes requires latex as it is used in matplotlib, we can instruct the LLM not to include it):
```bash
sudo apt update
sudo apt install -y \
  texlive-latex-base \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-fonts-extra \
  dvipng

```
#### verify it is installed:
```bash
latex --version
dvipng --version
```