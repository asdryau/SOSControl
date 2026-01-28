# **SOSControl: Enhancing Human Motion Generation Through Saliency-Aware Symbolic Orientation and Timing Control (AAAI 2026)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9.13](https://img.shields.io/badge/python-3.9.13-blue.svg)](https://www.python.org/downloads/release/python-3913/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-<2601.14258>-<COLOR>.svg)](https://arxiv.org/abs/2601.14258)
[![Supplementary](https://img.shields.io/badge/Supplementary%20Material-grey?style=flat&logo=Files&logoColor=white)](https://drive.google.com/file/d/1t-f0oKyfkU_MEslQkFVfFg8g9Hf45OD0/view?usp=sharing)

## 🎯 Abstract
<b>TL;DR</b>
> We present the SOS script and SOSControl framework for saliency-aware and precise control of body part orientation and motion timing in text-to-motion generation.

<details><summary><b>CLICK for full abstract</b></summary>

> Traditional text-to-motion frameworks often lack precise control, and existing approaches based on joint keyframe locations provide only positional guidance, making it challenging and unintuitive to specify body part orientations and motion timing. To address these limitations, we introduce the Salient Orientation Symbolic (SOS) script, a programmable symbolic framework for specifying body part orientations and motion timing at keyframes. We further propose an automatic SOS extraction pipeline that employs temporally-constrained agglomerative clustering for frame saliency detection and a Saliency-based Masking Scheme (SMS) to generate sparse, interpretable SOS scripts directly from motion data. Moreover, we present the SOSControl framework, which treats the available orientation symbols in the sparse SOS script as salient and prioritizes satisfying these constraints during motion generation. By incorporating SMS-based data augmentation and gradient-based iterative optimization, the framework enhances alignment with user-specified constraints. Additionally, it employs a ControlNet-based ACTOR-PAE Decoder to ensure smooth and natural motion outputs. Extensive experiments demonstrate that the SOS extraction pipeline generates human-interpretable scripts with symbolic annotations at salient keyframes, while the SOSControl framework outperforms existing baselines in motion quality, controllability, and generalizability with respect to motion timing and body part orientation control.
</details>

## 📚 Citation

If you find this work helpful in your research, please consider leaving a star ⭐️ and citing:

```bibtex
@inproceedings{au2026soscontrol,
  title={SOSControl: Enhancing Human Motion Generation Through Saliency-Aware Symbolic Orientation and Timing Control},
  author={Au, Ho Yin and Jiang, Junkun and Chen, Jie},
  year={2026},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}
}
```

## 📋 TODO

- ✅ Released model and dataloader code
- ✅ Released model checkpoints and data processing scripts
- ✅ Released code for generating evaluation motion samples
- 🔄 Provide demo script
- 🔄 Detailed instruction on running text-to-motion evaluation scripts in the external repository

## 🔮 Setup

### Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/asdryau/SOSControl.git
   cd SOSControl
   ```

2. **Create a conda environment**

   ```bash
   conda create -n soscontrol python=3.9.13
   conda activate soscontrol
   ```

3. **Install dependencies**

   ```bash
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

<!-- ## 📊 Data and Model Preparation -->

### Dataset and Pretrained Model

1. **Download**

   - Download `model_weights.zip` and `data.zip` from [HERE](https://drive.google.com/drive/folders/185UWJoTS-291ljy5iVbiP2kkeWLu3itY?usp=sharing)

2. **Repository Setup**

   - Extract both ZIP files and copy the contents into the `SOSControl/` directory of the current repository.

3. **File Structure**

   ```bash
   SOSControl
   ├── data
   │   ├──  hml3d_motion_data.pkl
   │   ├──  hml3d_split_data.pkl
   │   └──  hml3d_text_data.pkl
   ├── evaluation
   │   ├──  test_discLP_data.pkl
   │   └──  test_discLP_text.pkl
   └── model
       ├──  ControlDiffusion/lightning_logs/version_0/checkpoints/last.ckpt
       ├──  ControlPAE/lightning_logs/version_0/checkpoints/last.ckpt
       ├──  Diffusion/lightning_logs/version_0/checkpoints/last.ckpt
       └──  PAE/lightning_logs/version_0/checkpoints/last.ckpt
   ```
   
4. **Training Data Preprocessing**

   ```bash
   # process axis-angle and trans into 269-dim motion format
   python -m processed_data.process_data_format

   # extract SOS Scripts (before saliency thresholding)
   python -m processed_data.process_contLP
   python -m processed_data.process_discLP

   # process text using CLIP
   python -m processed_data.process_txtemb
   ```

## 🔧 Training

### 1. Train ACTOR-PAE

```bash
python -m model.PAE.train
```

### 2. Encode Training Data into Periodic Latent

```bash
python -m processed_data.process_paecode
```

### 2. Train Diffusion Model and ControlNets

```bash
# train model one by one
python -m model.Diffusion.train
python -m model.ControlDiffusion.train
python -m model.ControlPAE.train
```

## 📈 Evaluation
To generate the evaluation output for our model, execute the following commands:
```bash
python -m evaluation.test_diffuse
python -m evaluation.test_opt
```

To run the evaluation for the motion inbetweening task, execute the following commands:
```bash
python -m evaluation.evaluation_script
```

**Note:** Please refer to the [T2M Repository](https://github.com/EricGuo5513/text-to-motion) for details on the text-to-motion evaluation.

## 🖥️ Visualization
We use the SMPL-X Blender add-on to visualize the generated `.npz` file.

Please register at [(https://smpl-x.is.tue.mpg.de)](https://smpl-x.is.tue.mpg.de), download the SMPL-X for Blender add-on, and follow the provided installation instructions.

Once installed, select **Animation -> Add Animation** within the SMPL-X sidebar tool, and navigate to the generated `.npz` file for visualization.

## 🙏 Acknowledgments

- **[SMPL/SMPL-X](https://smpl.is.tue.mpg.de/)**: For human body modeling 
- **[PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/v0.3.0/pytorch3d/transforms/rotation_conversions.py)**: For rotation conversion utilities
- **[HumanML3D Dataset](https://github.com/EricGuo5513/HumanML3D)**: For motion and text data
- **[OmniControl](https://github.com/neu-vi/omnicontrol)**: For the HintBlock module in the ControlNet implementation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
