# Image Restoration and Diffusion Approaches for Removing Rain in Natural Scenes

This repository replicates the paper [**"Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning"**](https://arxiv.org/pdf/1904.05050). The framework addresses rain removal challenges by integrating a **physics-based model** and a **depth-guided conditional GAN**. This implementation restores clarity in images affected by heavy rain, improving visibility and detail preservation.

---

## Features

- **Two-Stage Framework**:
  - **Physics-Based Model**: Estimates and removes rain streaks, transmission maps, and atmospheric light.
  - **Conditional GAN**: Refines restored images for improved perceptual quality.
- Pre-trained model weights for direct use.
- Easy-to-use testing script for custom image restoration.
- Evaluation metrics like PSNR and SSIM for quantitative assessment.

---

## Repository Structure

```plaintext
deraining_in_natural_scenes/
│
├── calculate_psnr_ssim.py   # Computes PSNR and SSIM metrics
├── configs/                 # Configuration files for training and testing
├── data/                    # Directory for input data (images)
├── datasets/                # Custom dataset definitions
├── eval.py                  # Evaluation script
├── LICENSE                  # License file
├── models/                  # Model architectures
├── networks/                # Network definitions
├── README.md                # Project documentation
├── test.py                  # Script for testing images
├── train.py                 # Training script
├── train_diffusion.py       # Training script for diffusion models
└── utils/                   # Utility functions
```

## Download Pre-Trained Models
Download the pre-trained models from [Google Drive](https://drive.google.com/file/d/15EmQprHboSCjkqLSN17sFKc7UHDCWTTF/view?usp=sharing) and unzip the contents into the ckpt/ directory.

## Usage
To test the framework on your own images:
```
python test.py --input data/rainy_image.jpg --output out/restored_image.jpg
```

## Evaluate PSNR and SSIM
Use the evaluation script to compute PSNR and SSIM between restored and ground truth images:
```
python calculate_psnr_ssim.py --result <path_to_restored> --gt <path_to_ground_truth>
```

## Train the Model
To train the model from scratch:
```bash
python train.py --config configs/outdoorrain.yml
```
For training diffusion-based approaches:
```bash
python train_diffusion.py --config configs/outdoorrain.yml
```

## Potential Improvements
Future enhancements to the framework:

1. Integrate Diffusion Models for more effective rain removal.
2. Transformer-Based Architectures for improved global context handling.
3. Depth Estimation Independence to avoid reliance on depth maps.
4. Real-Time Optimization for applications in autonomous driving or surveillance.


## Citation
```cite
@article{zhang2019heavy,
  title={Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning},
  author={Zhang, et al.},
  journal={arXiv preprint arXiv:1904.05050},
  year={2019}
}
```