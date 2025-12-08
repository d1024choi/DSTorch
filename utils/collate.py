from utils.functions import *

def collate_Yolo(data):

    images, boxes, labels, start_end_idx = zip(*data)

    images = torch.stack(images, dim=0)
    boxes = torch.cat(boxes, dim=0)
    labels = torch.cat(labels, dim=0)
    start_end_idx = torch.LongTensor(torch.stack(start_end_idx, dim=0)) # batch x n_cams

    return {'images': images, 'boxes': boxes, 'labels': labels, 'start_end_idx': start_end_idx}
 