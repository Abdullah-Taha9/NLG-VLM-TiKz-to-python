"""LCEL chain for TikZ to Matplotlib conversion using Gemini multimodal."""

import re
import base64
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from PIL import Image

from config import GEMINI_API_KEY, MODEL_NAME


# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY
)

# Output parser
parser = StrOutputParser()

# System instruction
SYSTEM_INSTRUCTION = """You are an expert at converting TikZ code to Python Matplotlib code.

Given:
1. An image showing the desired visualization
2. A caption describing the image
3. The original TikZ code that generated the image

Your task: Generate clean, executable Python code using Matplotlib (and other necessary libraries like NumPy) that recreates the visualization as closely as possible.

Rules:
- Return ONLY the Python code, no explanations
- Include all necessary imports
- The code should be self-contained and runnable
- Use matplotlib.pyplot for plotting
- Match the visual style, colors, and layout of the original image
"""


def extract_code(response: str) -> str:
    """
    Extract clean Python code from markdown code blocks.
    
    Args:
        response: LLM response potentially wrapped in ```python ... ```
        
    Returns:
        Clean Python code without markdown wrappers
    """
    # Match ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\n?(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_prompt_content(image: Image.Image, caption: str, tikz_code: str) -> list:
    """
    Create multimodal prompt content for Gemini.
    
    Args:
        image: PIL Image of the visualization
        caption: Description of the image
        tikz_code: Original TikZ code
        
    Returns:
        List of content for HumanMessage
    """
    image_b64 = image_to_base64(image)
    
    return [
        {"type": "text", "text": SYSTEM_INSTRUCTION},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": f"\n**Caption:** {caption}\n\n**TikZ Code:**\n```latex\n{tikz_code}\n```\n\nPython code:"}
    ]


def invoke_chain(image: Image.Image, caption: str, tikz_code: str) -> str:
    """
    Invoke the LLM chain to convert TikZ to Matplotlib.
    
    Args:
        image: PIL Image of the visualization
        caption: Description of the image
        tikz_code: Original TikZ code
        
    Returns:
        Clean Python/Matplotlib code as string (markdown stripped)
    """
    content = create_prompt_content(image, caption, tikz_code)
    message = HumanMessage(content=content)
    
    # LCEL chain: message -> llm -> parser
    chain = llm | parser
    
    raw_response = chain.invoke([message])
    
    # Extract clean code from markdown wrappers
    return extract_code(raw_response)
