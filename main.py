# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 3 Jan, 2024

from model import MyDDPM, MyUNet, device, training_loop, loader, generate_new_images, save_images, save_first_batch
from torch.optim import Adam
import torch

batch_size = 128
n_epochs = 100
lr = 0.001
no_train = False

# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, 
              max_beta=max_beta, device=device)
print(sum([p.numel() for p in ddpm.parameters()]))

# store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
store_path = "ddpm_student_model.pt"
save_first_batch(loader)
if not no_train:
    training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)

best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")

print("Generating new images")
generated = generate_new_images(
        best_model,
        n_samples=9,
        device=device,
        #gif_name="fashion.gif" if fashion else "mnist.gif"
        gif_name="student.gif"
    )
save_images(generated, f"DDPM_Final result_{n_epochs}_epochs")