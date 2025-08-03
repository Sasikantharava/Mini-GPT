# app.py
import torch
import gradio as gr
from gpt import MiniGPT
from utils import load_data

# Load tokenizer and model
data, vocab_size, encode, decode = load_data("data/storytelling.txt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = MiniGPT(vocab_size)
model.load_state_dict(torch.load("gpt.pth", map_location=device))
model.to(device)
model.eval()

# Text generation function
@torch.no_grad()
def generate_text(prompt, max_new_tokens=200):
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return decode(idx[0].tolist())

# Gradio interface
interface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter a story prompt..."),
    outputs="text",
    title="MiniGPT Story Generator",
    description="Generate creative story completions with your own fine-tuned MiniGPT!"
)

interface.launch()
