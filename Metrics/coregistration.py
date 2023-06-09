import torch
import numpy as np
import torch.nn.functional as F
import warnings


warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths ")


def stacked_fineshift(img, shift_r, shift_c, device, sz=5):
    img = torch.clone(img).double()
    nbands = img.shape[1]
    kernel = torch.zeros(nbands, 1, sz, sz, device=device, dtype=img.dtype, requires_grad=False)

    if type(shift_r) == int:
        shift_r = [shift_r]*nbands
    if type(shift_c) == int:
        shift_c = [shift_c]*nbands
    if not torch.is_tensor(shift_r):
        shift_r = torch.tensor(shift_r, device=device, requires_grad=False)
    if not torch.is_tensor(shift_c):
        shift_c = torch.tensor(shift_c, device=device, requires_grad=False)

    r = shift_r
    c = shift_c

    r_int = r // 2
    c_int = c // 2

    r_frac = torch.remainder(r, 2)
    c_frac = torch.remainder(c, 2)

    condition = (r_frac == 1) * (c_frac == 1)
    if condition.count_nonzero() != 0:
        img[:, condition, :, :] = half_pixel_shift(img[:, condition, :, :], 'SE', half_interp23tap_kernel(condition.count_nonzero().item()), device)
    condition = (r_frac == 1) * (c_frac != 1)
    if condition.count_nonzero() != 0:
        img[:, condition, :, :] = half_pixel_shift(img[:, condition, :, :], 'S', half_interp23tap_kernel(condition.count_nonzero().item()), device)
    condition = (c_frac == 1) * (r_frac != 1)
    if condition.count_nonzero() != 0:
        img[:, condition, :, :] = half_pixel_shift(img[:, condition, :, :], 'E', half_interp23tap_kernel(condition.count_nonzero().item()), device)

    cnt = sz // 2
    b = torch.tensor(range(nbands), requires_grad=False).long()
    kernel[b, :, cnt - r_int, cnt - c_int] = 1

    shifted_img = F.conv2d(img, kernel, padding='same', groups=img.shape[1])

    return shifted_img



def half_interp23tap_kernel(nbands):
    halfKern = np.asarray([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0,
         0.000807762146, 0, -0.000060081482])
    halfKern = halfKern * 2.
    halfKern = np.repeat(halfKern[None, :], nbands, axis=0)
    halfKern = halfKern[:, None, None, :]
    return halfKern

def half_pixel_shift(img, direction, half_kernel):

    device = img.device
    img = img.double()
    batch_size, nbands, height, widht = img.shape

    directions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']
    assert direction in directions, "Error: wrong direction input '{}' - allowed values are ''N'', ''S'', ''NE'', etc.".format(
        direction)

    half_kernel = torch.from_numpy(half_kernel).to(device)


    kernel_x = torch.cat((torch.flip(half_kernel[:, :, :, 1:], dims=(-1,)), half_kernel), dim=3)
    kernel_x = kernel_x.transpose(0, 1)
    kernel_y = kernel_x.permute(1, 0, 3, 2)
    kernel_y = torch.flipud(kernel_y)
    pads = (kernel_y.shape[-1] // 2, kernel_y.shape[-1] // 2, kernel_y.shape[-2] // 2, kernel_y.shape[-2] // 2)
    kernel_xy = F.conv2d(F.pad(kernel_x, pads), kernel_y, padding='same', groups=nbands)

    kernel_x = kernel_x.transpose(0,1)
    kernel_xy = kernel_xy.transpose(0,1)
    kernel_x = kernel_x[:, :, :, ::2]
    kernel_y = kernel_y[:, :, ::2, :]
    kernel_xy = kernel_xy[:, :, ::2, ::2]


    if direction == 'N':
        h = kernel_y
    elif direction == 'S':
        h = torch.cat((kernel_y, torch.zeros(kernel_y.shape[0], kernel_y.shape[1], 1, kernel_y.shape[3], device=device)), dim=2)
    elif direction == 'W':
        h = kernel_x
    elif direction == 'E':
        h = torch.cat((kernel_x, torch.zeros(kernel_x.shape[0], kernel_x.shape[1], kernel_x.shape[2], 1, device=device)), dim=3)
    elif direction == 'NW':
        h = kernel_xy
    elif direction == 'NE':
        h = torch.cat((kernel_xy, torch.zeros(kernel_xy.shape[0], kernel_xy.shape[1], kernel_xy.shape[2], 1, device=device)), dim=3)
    elif direction == 'SW':
        h = torch.cat((kernel_xy, torch.zeros(kernel_xy.shape[0], kernel_xy.shape[1], 1, kernel_xy.shape[3], device=device)), dim=2)
    elif direction == 'SE':
        h = torch.cat((torch.cat((kernel_xy, torch.zeros(kernel_xy.shape[0], kernel_xy.shape[1], kernel_xy.shape[2], 1, device=device)), dim=3),
                           torch.zeros(kernel_xy.shape[0], kernel_xy.shape[1],  1, kernel_xy.shape[3] + 1, device=device)), dim=2)

    shifted_img = F.conv2d(img, h, padding='same', groups=nbands)

    return shifted_img






