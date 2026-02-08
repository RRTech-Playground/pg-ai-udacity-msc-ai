import torch

# On macOS, check if MPS is available
print(torch.backends.mps.is_available())

# On a machine with CUDA installed, check if CUDA is available
print(torch.cuda.is_available())

# Use MPS if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Again, the seme for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting the device testing mps, cuda and fallback to cpu - classic
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# or
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Modern Unified Accelerator API (PyTorch 2.4+)
# Automatically picks CUDA or MPS if available, otherwise defaults to CPU
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Device set to: {device}")

# Once the device variable is set, use the .to(device) method to move your models and tensors:
# Tensors: x = torch.randn(3, 3).to(device) or x = torch.ones(5, device=device)
# Models: model = MyModel().to(device)

# Critical M3 Note

# While the M3 is powerful, some operations may not be implemented for MPS yet.

# If you hit a NotImplementedError, you can force a fallback for those specific operations by setting this environment variable in your terminal before running your script:
# export PYTORCH_ENABLE_MPS_FALLBACK=1
# hm, IntelliJ also suggests following command, but not tested yet
# export TORCH_MPS_BACKEND=fallback

# Dtype Support: The MPS framework generally supports float32, but some versions may have limited support for float64.

# Single GPU: Unlike multi-GPU CUDA setups, MPS currently only supports a single GPU device.