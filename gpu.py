# For checking GPU availability

import torch
print(torch.cuda.is_available())  # Harus True jika CUDA bisa digunakan
print(torch.cuda.device_count())  # Jumlah GPU yang tersedia
print(torch.cuda.get_device_name(0))  # Nama GPU yang terdeteksi
