from sklearn.metrics import jaccard_score as jsc


def iou_from_tensors(pred_batch_tensor, label_batch_tensor, multi_class=False):  # Pred - (bs,21,h,w) ; label - (bs,h,w)
    pred_label_batch = pred_batch_tensor.argmax(dim=1).cpu().numpy().reshape(-1)
    label_batch = label_batch_tensor.cpu().numpy().reshape(-1)
    if multi_class:
        iou = jsc(pred_label_batch, label_batch, average='macro')
    else:
        iou = jsc(pred_label_batch, label_batch)
    return iou