from torchvision import transforms

def mnist_like(size=(32, 32)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),   # ONLY this
    ])

def cifar_to_gray(size=(32, 32)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size),
        transforms.ToTensor(),   # ONLY this
    ])
