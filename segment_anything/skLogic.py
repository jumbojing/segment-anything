### 医学影像预处理
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import interact

def readDcm(dcm_path):
    """
    读取DICOM图像序列,返回3D Image对象
    """
    # 读取DICOM图像序列
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)

    # 注意,此时还没有加载图像数据
    # 拼接3D图像
    image3d = reader.Execute()

    # 注意,如果图像维度<=2,则执行后image3d仍为None
    # 所以需要检验
    if image3d is None:
        print('Error: less than 3 dimensions')
        return None

    # 获得图像的元数据
    img_array = sitk.GetArrayFromImage(image3d)
    print('Image data type: ', image3d.GetPixelIDTypeAsString())
    print('Image size: ', image3d.GetSize())
    print('Image spacing: ', image3d.GetSpacing())
    print('Image origin: ', image3d.GetOrigin())
    print('Image dimension: ', image3d.GetDimension())

    return image3d

def readImgSk(filePath=None, img = None, img3C = True):
    '''sitk读图
    '''
    if img is None:
        # 若为文件夹
        # if os.path.isdir(filePath):
        if filePath.endswith('/'):
            img = readDcm(filePath)
        else:
            img = sitk.ReadImage(filePath)
    
    img_vbCT_255 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
    if img3C: # 三通道无法做魔法糖
        return sitk.GetArrayFromImage(sitk.Compose([img_vbCT_255]*3))
    else:
        return img_vbCT_255
def sk3C2gd(img):
    if img.GetDimension() == 3:
        return (sitk.VectorIndexSelectionCast(img, 0) +
                sitk.VectorIndexSelectionCast(img, 1) +
                sitk.VectorIndexSelectionCast(img, 2)) / 3

def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    if isinstance(img, np.ndarray):
        img = sitk.GetImageFromArray(img)
    size = img.GetSize()
    img_xslices = [img[s, :, :] for s in xslices]
    img_yslices = [img[:, s, :] for s in yslices]
    img_zslices = [img[:, :, s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_null = sitk.Image([0, 0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
        d += 1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen, d])
        # TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
            img = sitk.Compose(img_comps)

    myshow(img, title, margin, dpi)
    
    return img

def myshow(img, title=None, margin=0.05, dpi=80, cmap="gray"):
    if isinstance(img, np.ndarray):
        nda = np.copy(img)
        img = sitk.GetImageFromArray(img)
    else:
        nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            slicer = True

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if slicer:
        ysize = nda.shape[1]
        xsize = nda.shape[2]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    def callback(z=None):
        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        if z is None:
            ax.imshow(nda, extent=extent, interpolation=None, cmap=cmap)
        else:
            ax.imshow(nda[z, ...], extent=extent, interpolation=None, cmap=cmap)

        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0, nda.shape[0] - 1))
    else:
        callback()
