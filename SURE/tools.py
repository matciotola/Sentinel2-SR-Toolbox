import numpy as np
import torch
from torch.fft import fft2, ifft2

def gaussian_filter(N=15, sigma=2.0):
    n = (N - 1) / 2.0
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return torch.from_numpy(h)

def get_s2psf(band='band20', size =15):
    '''Get PSF of S2 bands, 20 m or 60 m'''
    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    mtf = np.array([.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23])
    sdf = d * np.sqrt(-2 * np.log(mtf) / np.pi ** 2)
    if band=='band20':
        sdf20 = sdf[d==2]
        psf20 = torch.zeros(len(sdf20),size,size)
        for i in range(len(sdf20)):
            psf20[i,:,:] = gaussian_filter(size,sdf20[i])
        return  psf20
    elif band=='band60':
        sdf60 = sdf[d == 6]
        psf60 = torch.zeros(len(sdf60), size, size)
        for i in range(len(sdf60)):
            psf60[i, :, :] = gaussian_filter(size, sdf60[i])
        return psf60
    elif band=='band10':
        sdf10 = sdf[d == 1]
        psf10 = torch.zeros(len(sdf10), size, size)
        for i in range(len(sdf10)):
            psf10[i, :, :] = gaussian_filter(size, sdf10[i])
        return psf10
    else:
        AssertionError


def expand_shift_psf(psf,c,m,n):
    '''pad psf to (mxn) with zeros, move the psf to the pad-center and shift--> reduce the boundary effect after convolution
    input: PSF (small size)
    output: PSF expand and shift to (c x m x n)'''
    midm=m//2
    midn=n//2
    if len(psf.shape)<3:
        midx=psf.shape[0]//2
        midy=psf.shape[1]//2
        y = torch.zeros(c, m, n)
    else:
        midx = psf.shape[1] // 2
        midy = psf.shape[2] // 2
        y = torch.zeros(psf.shape[0], m, n)
    y[:, midm - midx :midm + midx+1, midn - midy:midn + midy+1] = psf
    return torch.fft.fftshift(y,dim=[1,2])


def invAAtx(img, psf, ratio, cond):
    '''Compute inverse AAt applying to image x [(AAt)^-1x] which is equivalent to inverse 0th polyphase of BBt (blurring)[cite]
    Input: img: image (c x m x n), ratio, downsampling factor, h: PSF, cond: condition number
    output: inverse filtered image (c x m*ratio, n*ratio)'''
    c,m,n = img.shape
    device = psf.device
    x = img.to(device)
    nom = fft2(x)
    h = expand_shift_psf(psf, c, m*ratio, n*ratio).to(device)
    h0 = torch.real(ifft2(abs(fft2(h)) ** 2)) #BBT
    h0d = h0[:, ::ratio, ::ratio] #0 th component of the polyphase decomposition BBT
    aat = fft2(h0d)
    dem = aat+cond
    img_out = torch.real(ifft2(nom/dem))

    return img_out


def AAtx(img, psf, ratio):
    '''Compute AAt applying to image x [(AAt)^-1x] which is equivalent to 0th polyphase of BBt (blurring)[cite]
    Input: img: image (c x m x n), ratio, downsampling factor, h: PSF, cond: condition number
    output: inverse filtered image (c x m*ratio, n*ratio)'''
    c,m,n = img.shape
    device = psf.device
    x = img.to(device)
    nom = fft2(x)
    h = expand_shift_psf(psf, c, m*ratio, n*ratio).to(device)
    h0 = torch.real(ifft2(abs(fft2(h)) ** 2)) #BBT
    h0d = h0[:, ::ratio, ::ratio] #0 th component of the polyphase decomposition BBT
    aat = fft2(h0d)
    img_out = torch.real(ifft2(nom*aat))
    return img_out


def Atx(img,psf,ratio):
    '''upsampling by inserting zeros between samples and filtering
    input: img (c x m x n), ratio: upsampling ratio; psf: PSF()'''
    c, m, n = img.shape
    device = psf.device
    x = img.to(device)
    y = torch.zeros(c, m * ratio, n * ratio).to(device)
    y[:, ::ratio, ::ratio] = x # y=upsampling by inserting zeros
    h = expand_shift_psf(psf,y.shape[0],y.shape[1],y.shape[2]).to(device)
    img = torch.real(ifft2(fft2(y)*torch.conj(fft2(h))))
    return img


def Ax(img,psf,ratio):
    '''Filtering and downsampling by A
    input: image (c x m x n); psf: PSF (); ratio: downsampling ratio'''
    c,m,n = img.shape
    device = psf.device
    x = img.to(device)
    h = expand_shift_psf(psf,c,m,n).to(device)
    img = torch.real(ifft2(fft2(x)*fft2(h))) #filtering
    imgd = img[:,::ratio,::ratio] #downsampling
    return imgd


def back_projx(x,psf,ratio,cond):
    a = invAAtx(x,psf,ratio,cond)
    return Atx(a,psf,ratio)
