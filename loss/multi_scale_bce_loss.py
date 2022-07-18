from loss.bce_loss import cross_entropy_loss2d_sigmoid

def ms_bce_loss(out, labels, batch_size, model, side_weight, fuse_weight):
    bce_loss = 0
    if model == 'hed':
        for k in range(len(out)):
            bce_loss += cross_entropy_loss2d_sigmoid(out[k], labels)
        bce_loss = (bce_loss / len(out)) / batch_size
    elif model == 'bdcn':
        for k in range(10):
            bce_loss += side_weight*cross_entropy_loss2d_sigmoid(out[k], labels) / (10 * batch_size)
        bce_loss += fuse_weight*cross_entropy_loss2d_sigmoid(out[-1], labels) / batch_size
    return bce_loss
