# MoveCultureVision - Protótipo Conceitual
# Aprendizado visual autônomo com atualização de pesos em tempo real

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# --- 1) Definição do modelo ---
class AutoVision(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128):
        super(AutoVision, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, feature_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # Embeddings de imagem

# --- 2) Inicialização ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoVision().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Loss simplificada para protótipo

# --- 3) Pré-processamento de imagens ---
preprocess = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# --- 4) Função para treinar com nova imagem ---
def train_on_image(img_path):
    model.train()
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Auto-alvo: protótipo simples -> tentar reproduzir a própria imagem como vetor
    target = img_tensor.mean(dim=(2,3))  # Reduz a dimensão espacial, simplificação
    
    optimizer.zero_grad()
    output = model(img_tensor)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Treinado com {img_path}, Loss: {loss.item():.4f}")
    return output.detach().cpu().numpy()

# --- 5) Teste rápido ---
# output = train_on_image("sua_imagem_aqui.jpg")
# print("Embedding da imagem:", output)
