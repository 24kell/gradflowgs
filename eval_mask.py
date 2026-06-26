import os
import numpy as np
from PIL import Image
import cv2

gt_folder_path = os.path.join('')
pred_folder_path = os.path.join('')

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask trunblue_blue-office-desk  by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def load_mask(mask_path):
    """Load the mask from the given path."""
    # print(os.path.exists(mask_path))truck
    if os.path.exists(mask_path):
        return np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale
    else:
        return None

def resize_mask(mask, target_shape):
    """Resize the mask to the target shape."""
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))

def boundary_fscore(gt, dt, dilation_ratio=0.02):
    """
    Compute Boundary F-score between two binary masks.
    """

    gt = (gt > 128).astype(np.uint8)
    dt = (dt > 128).astype(np.uint8)

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)

    # 距离容忍范围
    h, w = gt.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    kernel = np.ones((3, 3), np.uint8)

    # 扩张边界用于匹配
    gt_dilate = cv2.dilate(gt_boundary, kernel, iterations=dilation)
    dt_dilate = cv2.dilate(dt_boundary, kernel, iterations=dilation)

    # precision: dt boundary 有多少被 gt 覆盖
    precision = (dt_boundary * gt_dilate).sum() / (dt_boundary.sum() + 1e-8)

    # recall: gt boundary 有多少被 dt 覆盖
    recall = (gt_boundary * dt_dilate).sum() / (gt_boundary.sum() + 1e-8)

    if precision + recall == 0:
        return 0.0

    fscore = 2 * precision * recall / (precision + recall)
    return fscore

def trimap_iou(gt, dt, dilation_ratio=0.02):
    gt = (gt > 128).astype('uint8')
    dt = (dt > 128).astype('uint8')

    h, w = gt.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    kernel = np.ones((3,3), np.uint8)
    
    gt_dilate = cv2.dilate(gt, kernel, iterations=dilation)
    gt_erode = cv2.erode(gt, kernel, iterations=dilation)

    # trimap区域（边界带）
    trimap = gt_dilate - gt_erode

    intersection = ((dt == gt) & (trimap == 1)).sum()
    union = (trimap == 1).sum()

    if union == 0:
        return 1.0
    
    return intersection / union

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = (dt > 128).astype('uint8')
    gt = (gt > 128).astype('uint8')

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()

    if union == 0:
        return 1.0  
    
    return intersection / union

def calculate_iou(mask1, mask2):
    """Calculate IoU between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_accuracy(mask1, mask2):
    """Calculate accuracy between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    correct_predictions = np.sum(mask1_bool == mask2_bool)
    total_pixels = mask1.size
    accuracy = correct_predictions / total_pixels
    return accuracy

if __name__ == "__main__":
    # 获取所有文件名
    gt_files = sorted(os.listdir(gt_folder_path))
    pred_files = sorted(os.listdir(pred_folder_path))
    print('gt :', len(gt_files), 'renders :', len(pred_files))

    ious = []
    bious = []
    tri_ious = []
    b_fscoles = []
    accuracies = []

    for idx in range(len(pred_files)):
        gt_file = os.path.join(gt_folder_path, gt_files[idx])
        pred_file = os.path.join(pred_folder_path, pred_files[idx])
        
        gt_mask = load_mask(gt_file)
        pred_mask = load_mask(pred_file)

        if gt_mask is not None and gt_mask.sum() != 0:
            # 计算IoU
            iou = calculate_iou(gt_mask, pred_mask)
            ious.append(iou)
            
            # 计算边界IoU
            biou = boundary_iou(gt_mask, pred_mask)
            bious.append(biou)
            
            # 计算 trimap-iou
            tri_iou = trimap_iou(gt_mask, pred_mask)
            tri_ious.append(tri_iou)
            
            # 计算准确率
            accuracy = calculate_accuracy(gt_mask, pred_mask)
            accuracies.append(accuracy)

            # 计算 boundary F-scole
            fscole = boundary_fscore(gt_mask, pred_mask)
            b_fscoles.append(fscole)

    # 计算平均值
    mIoU = np.mean(ious) if ious else 0
    mean_accuracy = np.mean(accuracies) if accuracies else 0
    mean_biou = np.mean(bious) if bious else 0
    mean_tri_iou = np.mean(tri_ious) if tri_ious else 0
    mean_fscole = np.mean(b_fscoles) if b_fscoles else 0

    print("mIoU :", mIoU)
    print("mAcc :", mean_accuracy)
    print("mBIoU: ", mean_biou)
    print("mtri-iou: ", mean_tri_iou)
    print("mfscole: ", mean_fscole)