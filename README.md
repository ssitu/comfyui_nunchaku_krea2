# comfyui_nunchaku_krea2

Run [Krea 2](https://huggingface.co/krea/Krea-2-Raw) in ComfyUI with
[Nunchaku](https://github.com/nunchux-ai/nunchaku) SVDQuant 4-bit models.

Seems to be slightly slower than int8 convrot quants, but uses much less VRAM which makes it faster on low VRAM setups. Your mileage may vary.

## Installation

1. Install [Nunchaku](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
   into ComfyUI's Python.
2. Clone this repository into `ComfyUI/custom_nodes`:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ssitu/comfyui_nunchaku_krea2
   ```

3. Restart ComfyUI.

## Usage

| File | Folder |
| --- | --- |
| A Krea 2 SVDQuant checkpoint, e.g. from [ssitu/Krea-2-Raw-SVDQuant](https://huggingface.co/ssitu/Krea-2-Raw-SVDQuant) | `models/diffusion_models` |
| [Qwen3-VL 4B text encoder](https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders) | `models/text_encoders` |
| [Qwen-Image VAE](https://huggingface.co/Comfy-Org/Krea-2/tree/main/vae) | `models/vae` |

Load the checkpoint with **Nunchaku Krea 2 DiT Loader**, the text encoder with `CLIPLoader`
(type `krea2`) and the VAE with `VAELoader`. LoRAs work with `LoraLoaderModelOnly`.

## Credits

- [Krea 2 Nunchaku port](https://github.com/sztlink/krea2-nunchaku)
- [Nunchaku](https://github.com/nunchux-ai/nunchaku) and [ComfyUI-nunchaku](https://github.com/nunchux-ai/ComfyUI-nunchaku)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
