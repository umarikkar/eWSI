import torch

def custom_collate_fn(batch):
    """
    Custom collate function to deal with lazy image loading.
    Each batch will be a generator that loads images in chunks of 128.
    """
    # Since we are using batch_size=1, we'll get a list of one item
    image_gen, label = batch[0]
    
    # Return the sample (which is a generator) directly
    return image_gen, torch.tensor([label]).half()