# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 3 Jan, 2024

from model import MyDDPM, MyUNet, device, training_loop, loader, generate_new_images, save_images, show_first_batch
from torch.optim import Adam
import torch

batch_size = 128
n_epochs = 2
lr = 0.001
no_train = False

# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, 
              max_beta=max_beta, device=device)
print(sum([p.numel() for p in ddpm.parameters()]))

# store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
store_path = "ddpm_student_model.pt"
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
save_images(generated, "Final result")











# Weird ChatGPT stuff below

# import torch
# from torchvision.utils import save_image
# from tqdm import tqdm
# import os
# from ddpm.ddpm import GaussianDDPM
# from ddpm.utils import inverse_sigmoid
# from PIL import Image
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader



# # Function to train the model
# def train_model(model, train_loader, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         for inputs, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=False):
#             optimizer.zero_grad()
#             loss = model(inputs)
#             loss.backward()
#             optimizer.step()

# # Function to generate new images
# def generate_images(model, num_images=10, output_dir='generated_images'):
#     model.eval()
#     os.makedirs(output_dir, exist_ok=True)
#     for i in range(num_images):
#         with torch.no_grad():
#             noise = torch.randn(1, 3, 256, 256)  # Adjust size according to your input size
#             output = model.sample(noise)
#             save_image(output, os.path.join(output_dir, f'image_{i+1}.png'))

# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.images = os.listdir(root_dir)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.images[idx])
#         image = Image.open(img_name).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image

# # Define transformation pipeline
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])


# # Main function
# def main():
#     # Load dataset and create DataLoader (Assuming you have your DataLoader implementation)
#     # ...
#     # Define model, optimizer
#     model = GaussianDDPM()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # Create custom dataset
#     dataset = CustomDataset(root_dir='sted_train_photos', transform=transform)

#     # Create DataLoader
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # Train the model
#     train_model(model, train_loader, optimizer, num_epochs=10)

#     # Generate new images
#     generate_images(model, num_images=10, output_dir='generated_images')

# if __name__ == "__main__":
#     main()