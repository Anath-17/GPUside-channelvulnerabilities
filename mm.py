import torch
import time

# Check GPU
device = "cuda"
print("GPU:", torch.cuda.get_device_name(0))

# Large matrices
N = 16384

A = torch.randn(N, N, device=device)
B = torch.randn(N, N, device=device)

# Warmup
for _ in range(10):
    C = torch.matmul(A, B)

torch.cuda.synchronize()

print("Starting benchmark...")

start = time.time()

for i in range(100):
    C = torch.matmul(A, B)

torch.cuda.synchronize()

end = time.time()

print(f"Elapsed time: {end-start:.2f} sec")