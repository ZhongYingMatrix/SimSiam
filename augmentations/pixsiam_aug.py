import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import GaussianBlur

    
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]


class PixSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur

        # get spatial trans record
        self.RandomResizedCrop = RandomResizedCropRecorded(image_size, scale=(0.2, 1.0))
        self.RandomHorizontalFlip = RandomHorizontalFlipRecorded()
        self.other_transform = T.Compose([
            # T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            # T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        x1, z1 = self.RandomResizedCrop(x)
        x1, f1 = self.RandomHorizontalFlip(x1)
        x1 = self.other_transform(x1)
        x2, z2 = self.RandomResizedCrop(x)
        x2, f2 = self.RandomHorizontalFlip(x2)
        x2 = self.other_transform(x2)
        # return x1, x2, z1, z2, f1, f2
        return (x1, z1, f1), (x2, z2, f2)


class RandomResizedCropRecorded(T.RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return (F.resized_crop(img, i, j, h, w, self.size, self.interpolation),
                torch.tensor([i, j, i+h, j+w]))


class RandomHorizontalFlipRecorded(T.RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return (F.hflip(img), True)
        return (img, False)


if __name__ == "__main__":
    from PIL import Image
    img = Image.open('tmp/test.jpg')
    #  img = Image.new('RGB', (32, 32), (255, 255, 255))
    t = PixSiamTransform(224)
    result = t(img)
    import torchvision.transforms as transforms
    for i in range(2):
        res_img = transforms.ToPILImage()(
            result[i][0]*torch.tensor(imagenet_mean_std[1]).reshape(3,1,1)+
            torch.tensor(imagenet_mean_std[0]).reshape(3,1,1))
        res_img.save('tmp/test_aug'+str(i)+'.jpg')
        print(i, result[i][1:])

