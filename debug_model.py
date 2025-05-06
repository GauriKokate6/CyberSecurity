import torch

# Load the checkpoint
checkpoint_path = "classifier.pth"  # Update this if your file name is different
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

print("\n🔍 Checking Model Weights from Checkpoint:")
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
