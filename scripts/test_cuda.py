import torch

print("torch version:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)

print("torch.cuda.is_available():", torch.cuda.is_available())
if not torch.cuda.is_available():
    quit()

print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"device {i}:", torch.cuda.get_device_name(i))
    x = torch.randn(1, device=f"cuda:{i}")
    print("allocated OK on", f"cuda:{i}")
