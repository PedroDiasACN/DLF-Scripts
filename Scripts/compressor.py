import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
import importlib

# 1. Configuring Absolute Directory Paths (Modify this to match your local setup)
PROJECT_ROOT = r"C:\Code\DLF\Dual-generative-Latent-Fusion"
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from entropy.compression_model import get_padding_size

def instantiate_from_config(config):
    module, cls = config["target"].rsplit(".", 1)
    cls_obj = getattr(importlib.import_module(module, package=None), cls)
    return cls_obj(**config.get("params", dict()))

def init_env():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

def main():
    init_env()
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Defined paths
    config_file = os.path.join(PROJECT_ROOT, "src", "config", "config_test.yaml")
    # Change "model_q0.ckpt" to q1, q2, or q3 depending on the desired compression level
    ckpt_path = os.path.join(PROJECT_ROOT, "Models", "model_q3.ckpt", "model_q3.ckpt") 
    # Your paths and directories may vary depending on your local setup, adjust them accordingly.
    
    dir_originals = os.path.join(PROJECT_ROOT, "Images", "Originals")
    dir_compressed = os.path.join(PROJECT_ROOT, "Images", "Compressed")
    os.makedirs(dir_compressed, exist_ok=True)

    # 2. Load the Model
    print("[INFO] Loading DLF model...")
    cfg = OmegaConf.load(config_file)
    cfg.model.params.ckpt_path = ckpt_path
    cfg.model.params.ignore_keys = ['epoch_for_strategy', 'lmbda_idx', 'lmbda_list']
    
    model = instantiate_from_config(cfg.model).to(DEVICE).eval()
    model.hybrid_codec.quantize_feat.force_zero_thres = 0.12
    model.hybrid_codec.quantize_feat.update(force=True)

    transform = transforms.Compose([transforms.ToTensor()])

    # 3. Compression Process
    for img_name in os.listdir(dir_originals):
        img_path = os.path.join(dir_originals, img_name)
        if not os.path.isfile(img_path):
            continue

        print(f"-> Compressing: {img_name}")
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        img_tensor = img_tensor * 2.0 - 1.0 # Normalization [-1, 1] required by the model

        im_H, im_W = img_tensor.shape[2], img_tensor.shape[3]
        
        # The model requires the dimensions to be multiples of 256
        pad_l, pad_r, pad_t, pad_b = get_padding_size(im_H, im_W, p=256)
        img_padded = torch.nn.functional.pad(
            img_tensor, (pad_l, pad_r, pad_t, pad_b), mode="replicate"
        )

        try:
            # Triggers the compiled C++ backend to generate the bitstream
            compress_output = model.compress(img_padded)
        except AttributeError:
            # Safe fallback: if the main class does not expose .compress() directly,
            # we extract the raw latent tensors from the encoder
            _, _, compress_output = model.encode_decode(img_padded, (im_H, im_W))

        # We pack the bits along with the crop information for decompression
        payload = {
            'data': compress_output,
            'original_shape': (im_H, im_W),
            'padding': (pad_l, pad_r, pad_t, pad_b)
        }

        # Save the compressed file (.dlf)
        base_name = os.path.splitext(img_name)[0]
        out_file = os.path.join(dir_compressed, f"{base_name}.dlf")
        torch.save(payload, out_file)
        
        print(f"   Saved to: {out_file}")
        
        # Free the VRAM
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()