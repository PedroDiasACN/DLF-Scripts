import os
import sys
import torch
from torchvision.utils import save_image
from omegaconf import OmegaConf
import importlib
import inspect
import traceback

# 1. Configuring Absolute Directory Paths (Modify this to match your local setup)
PROJECT_ROOT = r"C:\Code\DLF\Dual-generative-Latent-Fusion"
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

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
    
    # Defined paths (Modify these to match your local setup)
    config_file = os.path.join(PROJECT_ROOT, "src", "config", "config_test.yaml")
    ckpt_path = os.path.join(PROJECT_ROOT, "Models", "model_q3.ckpt", "model_q3.ckpt") 
    
    dir_compressed = os.path.join(PROJECT_ROOT, "Images", "Compressed")
    dir_decompressed = os.path.join(PROJECT_ROOT, "Images", "Decompressed")
    os.makedirs(dir_decompressed, exist_ok=True)

    # 2. Load the Model
    print("[INFO] Loading DLF model for decompression...")
    cfg = OmegaConf.load(config_file)
    cfg.model.params.ckpt_path = ckpt_path
    cfg.model.params.ignore_keys = ['epoch_for_strategy', 'lmbda_idx', 'lmbda_list']
    
    model = instantiate_from_config(cfg.model).to(DEVICE).eval()
    model.hybrid_codec.quantize_feat.force_zero_thres = 0.12
    model.hybrid_codec.quantize_feat.update(force=True)

    # Initialize the probability tables (CDFs)
    print("[INFO] Initializing entropy decoder probability tables...")
    if hasattr(model, 'set_torchac'):
        model.set_torchac()
    else:
        print("[Warning] Method set_torchac() not found. Decompression may fail.")

    # 3. Decompression Process
    for file_name in os.listdir(dir_compressed):
        if not file_name.endswith('.dlf'):
            continue
            
        file_path = os.path.join(dir_compressed, file_name)
        print(f"\n-> Decompressing: {file_name}")
        
        payload = torch.load(file_path, map_location=DEVICE, weights_only=False)
        pad_l, pad_r, pad_t, pad_b = payload['padding']
        dados = payload['data']
        
        try:
            img_rec_padded = model.decode_only(**dados)
            
        except TypeError:
            try:
                img_rec_padded = model.decode_only(dados)
            except Exception:
                print(f"[Fatal Error] The model rejected the data. Full log:")
                traceback.print_exc()
                break
                
        except Exception:
            print(f"[Fatal Error] Internal failure in the C++ model. Full log:")
            traceback.print_exc()
            break

        # Extract the raw tensor
        if isinstance(img_rec_padded, dict):
            img_rec_padded = img_rec_padded.get('x_hat', list(img_rec_padded.values())[0])
        elif isinstance(img_rec_padded, tuple):
            img_rec_padded = img_rec_padded[0]

        # Remove the padding
        img_rec = torch.nn.functional.pad(
            img_rec_padded, (-pad_l, -pad_r, -pad_t, -pad_b)
        )
        
        # Undo the normalization from [-1, 1] to [0, 1] and save
        img_rec = img_rec.clamp(-1.0, 1.0) / 2.0 + 0.5
        
        base_name = os.path.splitext(file_name)[0]
        out_file = os.path.join(dir_decompressed, f"{base_name}_reconstructed.png")
        save_image(img_rec, out_file)
        
        print(f"   [SUCCESS] Image saved to: {out_file}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()