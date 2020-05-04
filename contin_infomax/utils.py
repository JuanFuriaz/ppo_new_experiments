import warnings
import random
import numbers
import math
import torch
import torchvision


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndim):
        super(Unflatten, self).__init__()
        self.ndim = ndim

    def forward(self, x):
        return x.view(x.size(0), self.ndim, 1, 1)


class LinearClassifier(torch.nn.Module):
    def __init__(self, num_hidden, num_classes):
        super(LinearClassifier, self).__init__()
        self.map = torch.nn.Linear(num_hidden, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.apply(init_weights)

    def forward(self, x):
        return self.map(x)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss


class NonlinearClassifier(torch.nn.Module):
    def __init__(self, num_hidden, num_classes):
        super(NonlinearClassifier, self).__init__()
        self.map = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
            torch.nn.BatchNorm1d(num_features=num_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=num_hidden, out_features=num_classes, bias=True)
        )
        self.optimizer = torch.optim.Adam(self.parameters())
        self.apply(init_weights)

    def forward(self, x):
        return self.map(x)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss


class LinearRegressor(torch.nn.Module):
    def __init__(self, num_hidden):
        super(LinearRegressor, self).__init__()
        self.map = torch.nn.Linear(num_hidden, 1)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.apply(init_weights)

    def forward(self, x):
        return self.map(x)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x).view(-1)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss


class NonlinearRegressor(torch.nn.Module):
    def __init__(self, num_hidden):
        super(NonlinearRegressor, self).__init__()
        self.map = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
            torch.nn.BatchNorm1d(num_features=num_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=num_hidden, out_features=1, bias=True)
        )
        self.optimizer = torch.optim.Adam(self.parameters())
        self.apply(init_weights)

    def forward(self, x):
        return self.map(x)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x).view(-1)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss


def init_weights(self):
    # see also https://github.com/pytorch/pytorch/issues/18182
    for m in self.modules():
        if type(m) in {
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
            torch.nn.Linear,
        }:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                torch.nn.init.normal_(m.bias, -bound, bound)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         return_resized_erasure: boolean to return the erased and resized patch
            in addition to the erased image.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, return_resized_erasure=False, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.return_resized_erasure = return_resized_erasure
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
            patch = torchvision.transforms.functional.resized_crop(torchvision.transforms.functional.to_pil_image(img, "F"), x, y, h, w, size=img.shape[1:])
            img_new = torchvision.transforms.functional.erase(img, x, y, h, w, v, self.inplace)
            if self.return_resized_erasure:
                return img_new, torchvision.transforms.functional.to_tensor(patch)
            else:
                return img_new
        raise Exception("TODO: current code requires erasure")
        # return img
