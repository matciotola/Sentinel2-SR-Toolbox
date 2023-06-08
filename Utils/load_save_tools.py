from osgeo import gdal
import os
import numpy as np


def extract_info(filename):
    ds = gdal.Open(filename)
    geotransf = ds.GetGeoTransform()
    proj = ds.GetProjection()
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    tp = ds.GetRasterBand(1).DataType
    info = {'geo': geotransf, 'proj': proj, 'rows': rows, 'cols': cols, 'type': tp}
    del ds
    return info


def save_tiff(img, root, filename, info, ratio=2, type='float32', save_geo_info=True):

    if type == 'uint16':
        t = gdal.GDT_UInt16
        np.round(img).astype('uint16')
    elif type == 'float32':
        t = gdal.GDT_Float32
        img.astype('float32')

    if not os.path.exists(root):
        os.makedirs(root)
    r = ratio
    rows = info['rows']//r
    if not img.shape[0]==rows:
        img = np.moveaxis(img,0,-1)
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    b = img

    assert b.shape[0]==info['rows'] and b.shape[1]==info['cols'], \
        "The dimensions of the image and the reference are not the same."
    writer = driver.Create(os.path.join(root, filename), xsize=info['cols'],
                           ysize=info['rows'], bands=b.shape[-1],
                           eType=t)
    os.chmod(os.path.join(root, filename), 0o0777)
    for bb in range(b.shape[-1]):
        hband = writer.GetRasterBand(bb + 1)
        h = b[:, :, bb]
        h = h.astype(type)
        hband.WriteArray(h)
        # hband.SetNoDataValue(np.nan)
        # hband.FlushCache()
    geo = info['geo']
    proj = info['proj']
    geo2 = [0]*len(geo)

    if save_geo_info:
        writer.SetGeoTransform(geo)
        writer.SetProjection(proj)

    del driver
    del writer
    return
