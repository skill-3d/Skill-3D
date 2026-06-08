import gradio as gr
from paths import *

from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch
from inference import *
from utils import *

from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="ronormsigma1/dino_weight.pt", repo_type="model", cache_dir='./', resume_download=True)
print(ckpt_path)

save_path = './'
device = 'cpu'
dino = DINOv2_MLP(
                    dino_mode   = 'large',
                    in_dim      = 1024,
                    out_dim     = 360+180+360+2,
                    evaluate    = True,
                    mask_dino   = False,
                    frozen_back = False
                )

dino.eval()
print('model create')
dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
dino = dino.to(device)
print('weight loaded')
val_preprocess   = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')

def infer_func(img, do_rm_bkg, do_infer_aug):
    origin_img = Image.fromarray(img)
    if do_infer_aug:
        rm_bkg_img = background_preprocess(origin_img, True)
        angles = get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, device)
    else:
        rm_bkg_img = background_preprocess(origin_img, do_rm_bkg)
        angles = get_3angle(rm_bkg_img, dino, val_preprocess, device)
    
    phi   = np.radians(angles[0])
    theta = np.radians(angles[1])
    gamma = angles[2]
    confidence = float(angles[3])
    if confidence > 0.5:
        render_axis = render_3D_axis(phi, theta, gamma)
        res_img = overlay_images_with_scaling(render_axis, rm_bkg_img)
    else:
        res_img = img
    
    # axis_model = "axis.obj"
    return [res_img, round(float(angles[0]), 2), round(float(angles[1]), 2), round(float(angles[2]), 2), round(float(angles[3]), 2)]

server = gr.Interface(
    flagging_mode='never',
    fn=infer_func, 
    inputs=[
        gr.Image(height=512, width=512, label="upload your image"),
        gr.Checkbox(label="Remove Background", value=True),
        gr.Checkbox(label="Inference time augmentation", value=False)
    ], 
    outputs=[
        gr.Image(height=512, width=512, label="result image"),
        # gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model"),
        gr.Textbox(lines=1, label='Azimuth(0~360°)'),
        gr.Textbox(lines=1, label='Polar(-90~90°)'),
        gr.Textbox(lines=1, label='Rotation(-90~90°)'),
        gr.Textbox(lines=1, label='Confidence(0~1)')
    ]
)

server.launch()
