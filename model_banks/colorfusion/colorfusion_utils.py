
import torch
import numpy as np
import cv2
from colorfusion_archs.swin_arch import SwinIR
from torchvision.transforms import ToTensor

@torch.no_grad()
def gencolor_feed_data(img, hint_image, hint_mask=None, device=0):
    transform_img = ToTensor()
    img_tensor = transform_img(img.convert('RGB')).unsqueeze(0).to(device)
    hint_image_tensor = transform_img(hint_image.convert('RGB').resize(img.size)).unsqueeze(0).to(device)
    if hint_mask is not None:
        hint_mask = transform_img(hint_mask.convert('L')).unsqueeze(0).to(device)

    # mask img_color mask region to 0 
    data_dict = {'img_tensor': img_tensor, 'hint_image_tensor': hint_image_tensor, 'hint_mask': hint_mask}
    return data_dict

@torch.no_grad()
def gencolor_forward(model_fusion, data_tensor, return_np=True):
    if data_tensor['hint_mask'] is not None:
        x = torch.cat([data_tensor['img_tensor'], data_tensor['hint_image_tensor'], data_tensor['hint_mask']], dim=1)
    else:
        x = torch.cat([data_tensor['img_tensor'], data_tensor['hint_image_tensor']], dim=1)
    img_fusion = model_fusion(x)['pred']
    if return_np:
        img_fusion = tensor2img(img_fusion, rgb2bgr=False)
    return img_fusion

# baseline methods
def load_colorfusion_model(load_path, in_chans=6, device=0, model_type='base'):
    if model_type == 'base':
        model = SwinIR(in_chans=in_chans)
    elif model_type == 'small':
        model = SwinIR(in_chans=in_chans, window_size=4, depths=[6, 6], embed_dim=60, num_heads=[6, 6], mlp_ratio=2, upsampler='pixelshuffle')
    elif model_type == 'tiny':
        model = SwinIR(in_chans=in_chans, window_size=4, depths=[4], embed_dim=60, num_heads=[4], mlp_ratio=2, upsampler='pixelshuffle')
    elif model_type == 'depth6':
        model = SwinIR(in_chans=in_chans,depths=[6], embed_dim=60, num_heads=[6], mlp_ratio=2, upsampler='pixelshuffle')
    model.load_state_dict(torch.load(load_path, map_location='cpu')['params'], strict=True)
    model.eval()
    model = model.to(device)
    return model

def tensor2img(img_tensor, rgb2bgr=False):
    img = img_tensor.squeeze(0).detach().cpu().float().numpy() * 255
    img = np.transpose(img, (1, 2, 0))
    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

if __name__ == "__main__":
    load_colorfusion_model()