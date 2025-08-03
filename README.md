 🧠 Mini-GPT

Mini-GPT is a tiny GPT-style language model trained on a custom storytelling dataset using PyTorch.  
It includes training, text generation, and a Gradio-based user interface.

---

 📦 Features

- 🧠 Small-scale Transformer model built from scratch
- 📚 Train on any plain text file (`.txt`)
- 🖥️ Interactive text generation UI (via Gradio)
- 📊 Visual loss plot of training

---

 🛠️ Getting Started (VS Code or GitHub)

Whether you're working locally from scratch or cloning from GitHub, follow the steps below.

---

✅ Step-by-Step Setup

1. 🔁 Clone the repo or download manually

```bash
git clone https://github.com/Sasikantharava/Mini-GPT.git
cd Mini-GPT
2. 🧪 Create and activate virtual environment
bash
Copy code
 Create a virtual environment
python -m venv venv

 Activate it:
 On Windows:
venv\Scripts\activate
 On macOS/Linux:
source venv/bin/activate
3. 📦 Install required packages
bash
Copy code
pip install torch gradio matplotlib
Or if a requirements.txt is present:

bash
Copy code
pip install -r requirements.txt
4. 📝 Prepare your dataset
Edit or replace the file:

bash
Copy code
data/storytelling.txt
Make sure it’s a single .txt file with your story/stories.

5. 🏋️‍♂️ Train the model
Train your GPT model on the dataset:

bash
Copy code
python train.py --input_file data/storytelling.txt --output_file gpt.pth --max_iters 2000
You can change --max_iters or other hyperparameters as needed.

6. 📉 Check the training loss plot
After training, a graph will be saved as:

Copy code
loss_plot.png
This shows training and validation loss over time.

7. 🤖 Run the Gradio UI to generate text
bash
Copy code
python app.py
Then open your browser and go to:

cpp
Copy code
http://127.0.0.1:7860
8. ✍️ Generate text
Enter a starting phrase such as:

css
Copy code
Once upon a time
Click Submit and the model will complete it based on your training data.

📂 Project Structure
bash
Copy code
Mini-GPT/
├── app.py                  # Gradio web UI for generation
├── train.py                # Training script
├── generate.py             # Standalone generation script (CLI)
├── gpt.py                  # Model definition (MiniGPT)
├── utils.py                # Tokenizer and data handling
├── gpt.pth                 # Trained model file (after training)
├── loss_plot.png           # Training loss graph
├── data/
│   └── storytelling.txt    # Dataset file
├── .gradio/                # Gradio flagged examples (auto-created)
└── venv/                   # Virtual environment (optional)
