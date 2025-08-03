import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gpt import MiniGPT
from utils import load_data, get_batch

def main():
    # CLI Argument Parser
    parser = argparse.ArgumentParser(description="Train MiniGPT on a custom text dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to training .txt file")
    parser.add_argument("--output_file", type=str, default="gpt.pth", help="Path to save trained model")
    parser.add_argument("--max_iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding size")
    parser.add_argument("--block_size", type=int, default=64, help="Context window size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📦 Using device: {device.upper()}")

    # Load dataset
    data, vocab_size, encode, decode = load_data(args.input_file)
    train_data = data[:int(0.9 * len(data))]
    val_data = data[int(0.9 * len(data)):]

    # Initialize model
    model = MiniGPT(vocab_size, n_embd=args.n_embd, block_size=args.block_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    print("🚀 Starting training...")

    try:
        for step in range(args.max_iters):
            xb, yb = get_batch(train_data, args.block_size, args.batch_size)
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if step % 100 == 0 or step == args.max_iters - 1:
                with torch.no_grad():
                    xval, yval = get_batch(val_data, args.block_size, args.batch_size)
                    xval, yval = xval.to(device), yval.to(device)
                    val_logits = model(xval)
                    B, T, C = val_logits.shape
                    val_loss = F.cross_entropy(val_logits.view(B * T, C), yval.view(B * T))
                    val_losses.append(val_loss.item())
                    print(f"Step {step} | Train loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f}")
    except KeyboardInterrupt:
        print("🛑 Training interrupted. Saving model...")

    # Save loss curve
    plt.plot(train_losses, label="Train Loss")

    # Define validation steps
    steps = list(range(0, args.max_iters, 100))
    if args.max_iters - 1 not in steps:
        steps.append(args.max_iters - 1)

    # Align validation losses to steps
    min_len = min(len(steps), len(val_losses))
    plt.plot(steps[:min_len], val_losses[:min_len], label="Val Loss")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("MiniGPT Training Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    print("📉 Training loss plot saved as 'loss_plot.png'")

    # Save model
    torch.save(model.state_dict(), args.output_file)
    print(f"✅ Model saved to '{args.output_file}'")

if __name__ == "__main__":
    main()
