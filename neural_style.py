import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры нормализации VGG
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
VGG_STD  = torch.tensor([0.229, 0.224, 0.225]).to(device)

def load_cnn():
    """
    Предзагрузить VGG19 и вернуть (cnn, mean, std).
    """
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    return (cnn, VGG_MEAN, VGG_STD)

def image_loader(image_name, imsize=512):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imsave(tensor, filename):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image.clamp(0,1))
    image.save(filename)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0.0
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = 0.0
    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # mean/std должны быть тензорами размера (3,)
        self.mean = torch.tensor(mean).view(-1,1,1).to(device)
        self.std = torch.tensor(std).view(-1,1,1).to(device)
    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1','conv_2','conv_3','conv_4','conv_5']):
    cnn = cnn
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)
    content_losses = []
    style_losses = []

    i = 0  # индекс conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            name = f'layer_{i}'

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # убираем всё после последней loss
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j+1]

    return model, style_losses, content_losses

def run_style_transfer(content_img_path, style_img_path, output_path,
                       cnn_bundle=None, num_steps=60,
                       content_weight=1, style_weight=1e6,
                       imsize=640, progress_callback=None):
    """
    Выполнить NST. Если передан progress_callback(step, total), он будет вызван на каждой итерации.
    cnn_bundle: (cnn, mean, std) — если None, загрузит внутри.
    """
    content_img = image_loader(content_img_path, imsize=imsize)
    style_img   = image_loader(style_img_path, imsize=imsize)

    if cnn_bundle is None:
        cnn, mean, std = load_cnn()
    else:
        cnn, mean, std = cnn_bundle

    model, style_losses, content_losses = get_style_model_and_losses(cnn, mean, std, style_img, content_img)

    input_img = content_img.clone()
    # Параметризируем вход как оптимизируемый
    input_img.requires_grad_()

    optimizer = optim.LBFGS([input_img])

    run = [0]
    total = num_steps

    while run[0] < num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            if progress_callback is not None:
                try:
                    progress_callback(run[0], total)
                except Exception:
                    pass
            # вызываем callback после расчёта градиента (или до/после — на выбор)
            return style_score + content_score
        optimizer.step(closure)
        

    input_img.data.clamp_(0, 1)
    imsave(input_img, output_path)
