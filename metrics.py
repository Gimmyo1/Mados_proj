import numpy as np

def calculate_iou_per_class(y_true, y_pred, num_classes):
    
    iou_per_class = []
    for cls in range(num_classes):
       
        true_mask = (y_true == cls)
        pred_mask = (y_pred == cls)
        
      
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
      
        if union == 0:
            iou = 0.0  
        else:
            iou = intersection / union
        iou_per_class.append(iou)
    
    
    miou = np.mean(iou_per_class)
    
    return iou_per_class, miou