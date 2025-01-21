# Video-to-Video Few-Shot Patch-Based Training

A GAN-based image style transfer script that performs style transformation using traditional CNN approaches without relying on Diffusion models. This repository was created to achieve Video-to-Video transformation as an alternative to Text-to-Video approaches using Diffusion models.

The main objective is to perform stable style transfer through GAN-based V2V approach while considering frame-to-frame character movement consistency.

## Features

- 🎨 Style transfer using GAN architecture
- 📷 Sample size minimization through patch-based approach
- 🔲 Fast learning through convolution
- 🎞️ Improved learning accuracy with optional channel additions

## Usage

### Training
```bash
python run_training.py
```

### Generation
```bash
python generator.py
```

## Project Structure
```
├── src                # Program folder
├── config             # Configuration settings
├── output             # Training results
├── test_dataset       # Sample dataset
├── generator.py       # Inference processing
└── run_training.py    # Training execution
```

## Model Configuration
Please refer to the `config` folder for detailed settings.

## Future Implementation Plans (TBD)
- Training process optimization
- Inference process acceleration
- Integration of Vision Transformer (ViT) for Discriminator

## License
AGPL-3.0 license

## References
[1] O. Texler et al., [Interactive Video Stylization Using Few-Shot Patch-Based Training](https://arxiv.org/abs/2004.14489), 2004

[2] [OndrejTexler/Few-Shot-Patch-Based-Training](https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training)