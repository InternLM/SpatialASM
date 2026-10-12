<div align="center">
  <h1 align="center">SpatialASM: Internalizing Active Mentalization for Spatial Reasoning</h1>

  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=yQ8U3tUAAAAJ"><strong>Yuhong Liu<sup>*</sup></strong></a >
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=AC1sE1kAAAAJ"><strong>Shenxi Wu<sup>*</sup></strong></a >
    ·
    <a href="https://github.com/JunyiChi/"><strong>Junyi Chi</strong></a >
    .
    <a href="https://beichenzbc.github.io/"><strong>Beichen Zhang</strong></a >
    </br>
    <a href="https://lennoxdai.github.io/LennoxDai/"><strong>Xuanlang Dai</strong></a >
    ·
     <a href="https://myownskyw7.github.io/"><strong>Jiaqi Wang<sup>&dagger;</sup></strong></a >
    .
    <a href="https://yuhangzang.github.io/"><strong>Yuhang Zang<sup>&dagger;</sup></strong></a >
    ·
    <a href="http://dahua.site/"><strong>Dahua Lin<sup>&dagger;</sup></strong></a >    
  </p >
  <p align="center">
    <sup>*</sup>Equal contribution. &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding authors.
  </p>
  <p align="center" style="font-size: 1.2em; margin-top: 0.5em">
    📖<a href="http://arxiv.org/abs/2610.13832">Paper</a>
  | 🏠<a href="https://github.com/InternLM/Spatial-SSRL">Homepage</a >
  | 🤗<a href="https://huggingface.co/collections/YuhongLiu/spatialasm">SpatialASM Collection</a >
  | 🤗<a href="https://huggingface.co/collections/YuhongLiu/spatialasm">SpatialASM-128k Dataset</a >
  </p > 
</div>

## 📢 News

- 🔥 SpatialASM is coming soon. Please stay tuned for the paper, models, data, and code.

## 🌈 Overview

Humans rarely solve difficult spatial problems by passively inspecting the views in front of them. Instead, we revisit a scene from task-relevant viewpoints and mentally reconstruct what should be visible there. We call this capability **Active Spatial Mentalization (ASM)**.

We are thrilled to introduce **SpatialASM**, a scalable framework that internalizes ASM into LVLMs. Through the two-stage training on LVLMs, SpatialASM elicits active spatial exploration and imagination and enhances the overall spatial reasoning performance.
<p style="text-align: center;"> 
  <img src="assets/teaser.png" alt="Teaser" width="100%"> 
</p>

SpatialASM consists of three complementary components:

1. **Allocentric Novel-View Generator.** SpatialASM-NVG renders the scene from a target viewpoint specified by an anchor image, a pixel location, and a viewing direction. This allocentric formulation avoids lengthy sequences of camera-centric movements.
2. **ASM Trajectory Synthesis.** Given a spatial QA sample, an LVLM plans informative viewpoints and the NVG renders the corresponding observations. After GeoP and answer-consistency filtering, rendered views are replaced by textual captions to form explicit plan-imagine-reason trajectories for SFT.
3. **View-Imagination Reinforcement Learning.** Geometry-filtered candidate views are used to construct automatically verifiable perspective-taking tasks. GRPO further strengthens the association between a viewpoint description and its expected observation.
<p style="text-align: center;"> 
  <img src="assets/pipeline.png" alt="Pipeline" width="100%"> 
</p>


## 💡 Highlights

- 🔥 **Active spatial reasoning:** SpatialASM jointly internalizes task-conditioned **view planning** and viewpoint-conditioned **view imagination**, enabling LVLMs to spontaneously seek and mentally construct missing spatial evidence.
- 🔥 **Scalable spatial CoT supervision:** SpatialASM converts existing answer-only spatial QA data into explicit plan-imagine-reason trajectories and automatically verifiable perspective-taking tasks, alleviating the scarcity of spatial Chain-of-Thought supervision.
- 🔥 **Training-time spatial simulation:** A novel-view generator acts as temporary training scaffolding. The final model does not depend on a world model, renderer, or additional visual observations at inference time.
- 🔥 **Grounded and reliable supervision:** An allocentric viewpoint interface reaches a target perspective in one query, while the **Geometry Perturbation (GeoP)** metric and answer-consistency checks filter unreliable synthesized views and trajectories.
<p style="text-align: center;"> 
  <img src="assets/comparison.png" alt="Comparison" width="100%"> 
</p>


## 📊 Results

We train four LVLM backbones of Qwen3-VL and Qwen3.5 with our SpatialASM framework, and the experimental results on both spatial reasoning and general-VQA tasks are shown below.

<p style="text-align: center;"> 
  <img src="assets/results.png" alt="Exp" width="100%"> 
</p>

## ✒️Citation
If you find this project useful, please kindly cite:
```
@article{liu2026spatialasm,
  title={SpatialASM: Internalizing Active Mentalization for Spatial Reasoning},
  author={Yuhong Liu, Shenxi Wu, Junyi Chi, Beichen Zhang, Xuanlang Dai, Jiaqi Wang, Yuhang Zang, Dahua Lin},
  journal={arXiv preprint arXiv:2610.13832},
  year={2026}
}
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) 

**Use of SpatialASM-NVG**: The model <a href="https://huggingface.co/YuhongLiu/SpatialASM-NVG">SpatialASM-NVG</a > is trained on <a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-9B">FLUX.2-klein-9B</a >, so please follow <a href="https://github.com/black-forest-labs/flux2/blob/main/model_licenses/LICENSE-FLUX-NON-COMMERICAL">FLUX-NON-COMMERCIAL LICENSE</a >.

**Usage and License Notices**: The data and code are intended and licensed for research use only.

## ❤️ Acknowledgement
The evaluation part is built on <a href="https://github.com/open-compass/VLMEvalKit">VLMEvalkit</a >, a powerful toolkit to evaluate a vast range of LMMs!

The training part is built on <a href="https://github.com/modelscope/DiffSynth-Studio">DiffSynth-Studio</a >, an excellent codebase for Diffusion model training!
