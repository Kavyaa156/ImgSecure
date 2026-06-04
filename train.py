import torch
import os

print("---Starting Training Simulation---")
print(f"Training on GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU")

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "model_weights.txt"), "w") as f:
    f.write("Epoch 10: Loss = 0.023\nOptimization Complete.")

print("Saved dummy weights to outputs/model_weights.txt")
