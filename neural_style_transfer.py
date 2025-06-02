import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import psutil

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper to check memory usage
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

# Helper to load and transform image
def load_image(path, size=64):
    """
    Load and preprocess an image.
    
    Args:
        path (str): Path to the image
        size (int): Size to resize the image (square)
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if not isinstance(path, str):
        raise ValueError(f"Image path must be a string, got {type(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    
    try:
        image = Image.open(path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(device, torch.float)
        print(f"Loaded image {path} with shape {image.shape}")
        print_memory_usage()
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image {path}: {str(e)}")

# Helper to display image
def show_image(tensor, title=None):
    """
    Display a tensor as an image.
    
    Args:
        tensor (torch.Tensor): Image tensor
        title (str, optional): Title for the plot
    """
    try:
        image = tensor.cpu().clone().squeeze(0)
        denorm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        image = denorm(image).clamp(0, 1)
        image = transforms.ToPILImage()(image)
        if title:
            print(title)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {str(e)}")

# Model wrapper to extract style/content features
class StyleContentModel(nn.Module):
    def __init__(self, model, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.model = model
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.features = []
        self.layer_names = []
        i = 1
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                name = f'conv_{i}'
                i += 1
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i-1}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i-1}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i-1}'
            else:
                continue
            self.features.append(layer)
            self.layer_names.append(name)
        self.model = nn.Sequential(*self.features)

    def forward(self, x):
        style_feats = []
        content_feats = []
        for name, layer in zip(self.layer_names, self.model):
            x = layer(x)
            if name in self.style_layers:
                style_feats.append(x)
            if name in self.content_layers:
                content_feats.append(x)
        return style_feats, content_feats

# Gram matrix for style loss
def gram_matrix(tensor):
    """
    Compute the Gram matrix for style loss.
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Gram matrix
    """
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

def main():
    # Define layers for content and style
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    # Image paths (replace with actual paths)
    content_path = "content_image.jpg"  # E.g., a landscape photo
    style_path = "style_image.jpg"  # E.g., a starry night painting
    
    try:
        # Verify paths
        print(f"Checking content path: {content_path}")
        print(f"Checking style path: {style_path}")
        if not os.path.exists(content_path):
            raise FileNotFoundError(f"Content image not found at {content_path}")
        if not os.path.exists(style_path):
            raise FileNotFoundError(f"Style image not found at {style_path}")
        
        # Load images
        print("Loading images...")
        content = load_image(content_path, size=64)
        style = load_image(style_path, size=64)
        
        # Load VGG19 model
        print("Loading VGG19 model...")
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
        print_memory_usage()
        
        # Initialize model
        print("Initializing style/content model...")
        model = StyleContentModel(vgg, style_layers, content_layers).to(device)
        
        # Extract features
        print("Extracting style and content features...")
        style_feats, _ = model(style)
        _, content_feats = model(content)
        
        # Optimize target image
        target = content.clone().requires_grad_(True).to(device)
        optimizer = optim.Adam([target], lr=0.01)
        
        # Weights
        style_weight = 1e6
        content_weight = 1.0
        
        # Optimization loop
        steps = 300
        print("Starting optimization...")
        for step in range(steps):
            try:
                target_style_feats, target_content_feats = model(target)
                style_loss = 0
                content_loss = 0
                
                # Style loss
                for ts, ss in zip(target_style_feats, style_feats):
                    style_loss += torch.mean((gram_matrix(ts) - gram_matrix(ss)) ** 2)
                
                # Content loss
                for tc, cc in zip(target_content_feats, content_feats):
                    content_loss += torch.mean((tc - cc) ** 2)
                
                # Total loss
                loss = style_weight * style_loss + content_weight * content_loss
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)  # Added as precaution
                optimizer.step()
                
                if step % 50 == 0 or step == steps - 1:
                    print(f"Step [{step}/{steps}] complete, Loss: {loss.item():.4f}")
                    print_memory_usage()
            except Exception as e:
                print(f"Optimization failed at step {step}: {str(e)}")
                raise
        
        # Show and save final result
        print("Saving and displaying stylized image...")
        show_image(target, title="Stylized Image")
        output_path = "stylized_image.jpg"
        output_image = target.cpu().clone().squeeze(0)
        denorm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        output_image = denorm(output_image).clamp(0, 1)
        output_image = transforms.ToPILImage()(output_image)
        output_image.save(output_path)
        print(f"Output saved to {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please provide valid image paths.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to process images; {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
