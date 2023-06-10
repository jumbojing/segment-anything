### 医学影像预处理
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import interact
import pickle
import supervision as sv
import itertools as itt

def pk2file(file, data=None):
    if not file.endswith('.pickle'):
        file = file+'.pickle'    
    if data is not None:
        with open(file, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

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
    
def cropOtsu(image):
    ''' Otsu阈值裁图
    '''
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter() # 用于计算图像的轴对齐边界框。
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value) # 获取边界框
    return sitk.RegionOfInterest(image,
                                 bounding_box[int(len(bounding_box) / 2) :],
                                 bounding_box[0 : int(len(bounding_box) / 2)],)

def sk3C2gd(img):
    # 三通道转灰度
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

    skShow(img, title, margin, dpi)
    
    return img

def orientDic(img):
    return {'X': 'L-R'[img.GetDirection()[0]+1],
            'Y': 'P-A'[img.GetDirection()[4]+1],
            'Z': 'I-S'[img.GetDirection()[8]+1]}

from ipywidgets import interact
import copy
def skShow(img,
           slice = 'Z',
           title=None,
           margin=0.05,
           dpi=80,
           fSize=None,
           cmap="gray"):
    if isinstance(img, np.ndarray):
        nda = np.copy(img)
        img = sitk.GetImageFromArray(img)
    else:
        img = copy.deepcopy(img)
        nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    size = img.GetSize()
    if nda.ndim == 3: # 若3维数组
        # fastest dim, either component or x
        c = nda.shape[-1] # 通道数
        if c in (3, 4): # 若通道数为3或4, 则认为是2D图像
            nda = nda[:,:,0]
    elif nda.ndim == 4: # 若4维数组
        c = nda.shape[-1]
        if not c in (3, 4): # 若通道数不为3或4, 则认为是3Dv(4D)图像, 退出
            raise RuntimeError("Unable to show 3D-vector Image")
        else:
            # 去掉最后一维
            nda = nda[:,:,:,0]
    if nda.ndim == 2: # 若2维数组
        nda = nda[np.newaxis, ...] # nda增加后的维度为3维, 且最后一维为1
        size = size + (1,) # size增加后的维度为3维, 且最后一维为1
        spacing = 1.
    # nda.shape = shape# nda的方向为LPS
    # size = nda.shape # size为z,y,x
    print('size:',size)
    xyzSize = [int(i+1)
               for i
               in (np.array(spacing)*np.array(size))
              ]
    sInd = dict(Z=0, Y=1, X=2)
    sDic = [dict(drt=['A-->P', 'L<--R'],
                 nda=nda, # nda的方向为LP
                 x = xyzSize[0],
                 y = xyzSize[1],
                 z = size[2],
                 extent = (xyzSize[0],0,0,xyzSize[1]) # (left, right, bottom, top)
                 ),
            dict(drt=['I<--S', 'L<--R'],
                 nda=np.transpose(nda, (1,0,2)), # nda的方向为LS
                 x = xyzSize[0],
                 y = xyzSize[2],
                 z= size[1],
                 extent = (xyzSize[0],0,xyzSize[2],0) # (left, right, bottom, top)
                 ),
            dict(drt=['I<--S', 'P<--A'],
                 nda=np.transpose(nda, (2,0,1)), # nda的方向为SP
                 x = xyzSize[1],
                 y = xyzSize[2],
                 z = size[0],
                 extent = (xyzSize[1],0,xyzSize[2], 0) # (left, right, bottom, top)
                 )][sInd[slice]]
    if fSize is not None:
        figsize = fSize
    else:
        figsize = (1 + margin) * sDic['x'] / dpi, (1 + margin) * sDic['y'] / dpi # TypeError: string indices must be integers
    # extent = (0, sDic['xyzSize'][0], sDic['xyzSize'][1],0) # (left, right, bottom, top)
    nda = sDic['nda']
    drt = sDic['drt']
    def callback(axe=None):

        fig = plt.figure(figsize=figsize, dpi=dpi) 

        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        ax.imshow(nda[axe, ...], extent=sDic['extent'], interpolation=None, cmap=cmap)

        # 在图像上标注坐标轴
        ax.set_ylabel(drt[0])
        ax.set_xlabel(drt[1])

        if title:
            plt.title(title)

        return plt.show()
    interact(callback, axe=(0, sDic['z'] - 1))
    
    
def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    xyxy = boxes_xywh.copy()
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    return xyxy


def svAllSam(sData, cId=0): # 将全分的结果转为sv.Dets
    
    if isinstance(sData[0], dict):
        sorted_generated_masks = sorted(
            sData, key=lambda x: x["area"], reverse=True
        )
        xywh = np.array([mask["bbox"] for mask in sorted_generated_masks])
        mask = np.array([mask["segmentation"] for mask in sorted_generated_masks])
        confidence = np.array([mask["predicted_iou"] for mask in sorted_generated_masks])
        tracker_id = np.arange(len(sorted_generated_masks))
        class_id = np.ones(len(sorted_generated_masks))*cId
        return sv.Detections(xyxy=xywh_to_xyxy(xywh),
                            mask=mask,
                            confidence=np.array(confidence),
                            tracker_id=tracker_id,
                            class_id=class_id
                            )
    elif isinstance(sData[0], list):
        for i, data in enumerate(sData):
            svData = svAllSam(data, cId=i)
            if i == 0:
                dets = sv.Detections(xyxy=svData.xyxy,
                                    mask=svData.mask,
                                    confidence=svData.confidence,
                                    tracker_id=svData.tracker_id,
                                    class_id=svData.class_id
                                    )

            else:
                dets.xyxy=np.append(dets.xyxy,svData.xyxy, axis=0)
                dets.mask=np.append(dets.mask,svData.mask, axis=0)
                dets.confidence=np.append(dets.confidence,svData.confidence, axis=0)
                dets.tracker_id=np.append(dets.tracker_id,svData.tracker_id, axis=0)
                dets.class_id=np.append(dets.class_id,svData.class_id, axis=0)
        return dets

def svShow(img, anns = None, dets=None, cId=None, tId=None, mask=True, svBx=True, show=False):
    ''' sv显示万分的anns
    
    '''
    if anns is not None:
        dets = svAllSam(anns)
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    if cId is None:
        cId = dets.class_id
    if tId is None:
        tId = dets.tracker_id
    labels = []
    for _, _, confidence, class_id,_ in dets:
        label = f'{cId}_{tId}: {confidence:0.2f}'
        labels.append(label)    
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    if mask:
        annotated_image = mask_annotator.annotate(scene=img, detections=dets)
        if svBx:
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=dets, labels=labels)
    else:
        if svBx:
            annotated_image = box_annotator.annotate(scene=img, detections=dets, labels=labels)
        else:
            print('啥都不要, 所以啥都没有...')
            return  dets     
    if show:
#         %matplotlib inline
        sv.plot_image(annotated_image, (16, 16))
    else:
        return annotated_image, dets

    
def getBbox(img, thr=0, crop=False, pad=0):
    """获取2D和3D图像的bbox, 返回xyxy或xyzxyz"""
    if isinstance(img, np.ndarray):
        img = sitk.GetImageFromArray(img)
    img = img > thr
    d = img.GetDimension()
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(sitk.Cast(img, sitk.sitkUInt8))
    bb = np.array(ls.GetBoundingBox(1))
    ini, size = bb[:d], bb[d:]
    fin = ini + size
    ini -= pad
    fin += pad
    image_size = np.array(img.GetSize())
    ini[ini < 0] = 0 # 防止ini小于0
    for i in range(d):
        fin[i] = min(fin[i], image_size[i]) # 防止fin超出图像的范围
    size = fin - ini
    if crop:
        if d == 2:
            return img[ini[0]:fin[0], ini[1]:fin[1]]
        else:
            return img[ini[0]:fin[0], ini[1]:fin[1], ini[2]:fin[2]]
    else:
        return np.array(ini.tolist() + fin.tolist())    
    
    # The function which is used to rotate (and make the 3D image isotropic) using euler angles
# https://discourse.itk.org/t/simpleitk-euler3d-transform-problem-with-output-size-resampling/4387/4
def skRot(image, angs = [0,0,0], opSpc = None, bgVal=0.0):

    euTfm = sitk.Euler3DTransform (
                image.TransformContinuousIndexToPhysicalPoint(
                    [(sz-1)/2.0 for sz in image.GetSize()]),
                np.deg2rad(angs[0]),
                np.deg2rad(angs[1]),
                np.deg2rad(angs[2]))

    # compute the resampling grid for the transformed image
    max_indexes = [sz-1 for sz in image.GetSize()]
    exInds = list(itt.product(*(
                list(zip([0]*image.GetDimension(),
                        max_indexes)))))
    exPsTfmd = [euTfm.TransformPoint(
                    image.TransformContinuousIndexToPhysicalPoint(
                        p))
                        for p in exInds]

    opMinCds = np.min(exPsTfmd, axis=0)
    opMaxCds = np.max(exPsTfmd, axis=0)

    # isotropic ouput spacing
    if opSpc is None:
      opSpc = min(image.GetSpacing())
    opSpc = [opSpc]*image.GetDimension()

    opOri = opMinCds
    opSize = [int(((omx-omn)/ospc)+0.5)
              for ospc, omn, omx
              in zip(opSpc, opMinCds, opMaxCds)]

    opDrt = [1,0,0,0,1,0,0,0,1]
    opPxTp = image.GetPixelIDValue()

    return sitk.Resample(image,
                         opSize,
                         euTfm.GetInverse(),
                         sitk.sitkLinear,
                         opOri,
                         opSpc,
                         opDrt,
                         bgVal,
                         opPxTp)
