import torch
from sklearn.metrics import jaccard_score as jsc

def get_label_names():
    '''
    Label info --
        0: Background
        1-20: Objects,
        255: Null (at the borders and on difficult objects)
    '''
    label_mapping = {0:'BACKGROUND',
                     1:'Aeroplane', 2:'Bicycle', 3:'Bird', 4:'Boat', 5:'Bottle',
                     6:'Bus', 7:'Car', 8:'Cat', 9:'Chair', 10:'Cow',
                     11:'Dining table', 12:'Dog', 13:'Horse', 14:'Motor bike', 15:'Person',
                     16:'Potted plant', 17:'Sheep', 18:'Sofa', 19:'Train', 20:'TV monitor',
                     255:'NULL'}

    return label_mapping


def label2rgb(label_mask_pil):
    rgb_mask_pil = label_mask_pil.convert('RGB')
    return rgb_mask_pil


def label2onehot(label_batch_tensor):
    n_classes = 21
    batch_size = label_batch_tensor.shape[0]
    
    oh_label_batch = torch.zeros((batch_size, label_batch_tensor.shape[1], label_batch_tensor.shape[2], n_classes)).int()
    
    for c in range(n_classes):
        oh_label_batch[:,:,:,c][label_batch == c] = 1
    oh_label_batch = oh_label_batch.permute((0,3,1,2))
    
    return oh_label_batch


def normalize_intensities(image_batch_tensor, normalization='min-max'):
    if normalization == 'min-max':
        image_batch_tensor = image_batch_tensor.float() / 255 

    if normalization == 'z-score':
        batch_mean = torch.mean(image_batch_tensor) # Single mean value over the entire batch and all channels
        batch_stddev = torch.std(image_batch_tensor)
        image_batch_tensor = (image_batch_tensor.float() - batch_mean) / batch_stddev
    
    return image_batch_tensor


def iou_from_tensors(pred_batch_tensor, label_batch_tensor):  # Pred - (bs,21,h,w) ; label - (bs,h,w)
    pred_label_batch = pred_batch_tensor.argmax(dim=1).cpu().numpy().reshape(-1)
    label_batch = label_batch_tensor.cpu().numpy().reshape(-1)
    iou = jsc(pred_label_batch, label_batch, average='macro')
    return iou