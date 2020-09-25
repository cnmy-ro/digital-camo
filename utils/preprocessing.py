import torch



def normalize_intensities(image_batch_tensor, normalization='min-max'):
    if normalization == 'min-max':
        image_batch_tensor = image_batch_tensor.float() / 255 

    if normalization == 'z-score':
        batch_mean = torch.mean(image_batch_tensor) # Single mean value over the entire batch and all channels
        batch_stddev = torch.std(image_batch_tensor)
        image_batch_tensor = (image_batch_tensor.float() - batch_mean) / batch_stddev
    
    return image_batch_tensor