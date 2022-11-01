def neg_iou_loss(predict, target):
    dims = tuple(range(len(predict.shape))[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.numel()