import torch
print(f"CUDA disponible : {torch.cuda.is_available()}")
print(f"Device count : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")