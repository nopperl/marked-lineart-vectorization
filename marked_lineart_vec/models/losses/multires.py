from kornia.geometry.transform import PyrDown

dsample = PyrDown()


def gaussian_pyramid_loss(recons, input_img, loss_fn, seg_loss_fn, seg_loss_weight=1):
    recon_loss = loss_fn(recons, input_img, reduction="none").mean(dim=[1, 2, 3]) + seg_loss_weight * seg_loss_fn(recons, input_img)
    for j in range(2, 5):
        recons = dsample(recons)
        input_img = dsample(input_img)
        recon_loss = recon_loss + loss_fn(recons, input_img, reduction="none").mean(dim=[1, 2, 3]) + seg_loss_weight * seg_loss_fn(recons, input_img)
    return recon_loss