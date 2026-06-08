<h1 align="center">World Pilot: Steering Vision-Language-Action Models with World-Action Priors</h1>

<p align="center">Zefu Lin, Rongxu Cui, Junjia Xu, Xiaojuan Jin, Wenling Li, Lue Fan, and Zhaoxiang Zhang</p>

<p align="center">
  <a href="https://world-pilot.github.io/"><img alt="Project Page" src="https://img.shields.io/static/v1?label=&message=Project%20Page&color=111827&style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="#"><img alt="arXiv" src="https://img.shields.io/static/v1?label=&message=arXiv&color=B31B1B&style=flat-square&logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/Chedan86/WorldPilot-LIBERO"><img alt="Hugging Face" src="https://img.shields.io/static/v1?label=&message=Hugging%20Face&color=F4B400&style=flat-square&logo=huggingface&logoColor=111827"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#news">News</a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#documentation">Documentation</a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#citation">Citation</a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#acknowledgements">Acknowledgements</a>
</p>

<a id="overview"></a>
## Overview

<p align="center">
  <img src="./assets/teaser.png" alt="WorldPilot teaser" width="100%" />
</p>

<p>
  World Pilot steers a VLA with priors from a World-Action Model. VLA methods generate actions from a
  VLM's encoding of the scene. World Pilot adds two priors from a WAM into the decision chain, with
  Latent Steering routing a scene-evolution latent into VLM hidden states and Action Steering feeding a
  trajectory-level motion prior to the action generator. This gives the VLA an anticipated view of the
  scene and a motion hint alongside its semantic conditioning. World Pilot reaches state-of-the-art
  performance on LIBERO-Plus and real-robot tasks.
</p>

<a id="news"></a>
## ✨ News

- **[2026.6.9]** WorldPilot model weights are now available on Hugging Face. Feel free to try them out! 🚀
- **[2026.6.8]** The WorldPilot codebase is released, including training and evaluation. Model weights are coming soon. 🚀
- **[2026.6.7]** WorldPilot is now live on arXiv. The code is coming soon. 🚀

<a id="documentation"></a>
## 📚 Documentation

The documentation below covers environment setup, training, and public evaluation. Start with the
installation guide, then follow the training or evaluation notes for your workflow.

- [Installation](./doc/environment.md)
- [Training](./doc/training.md)
- [Evaluation](./doc/evaluation.md)

## 🤗 Model Zoo

We release our pretrained model parameters and precomputed LIBERO cache on Hugging Face.

- Model weights: [Chedan86/WorldPilot-LIBERO](https://huggingface.co/Chedan86/WorldPilot-LIBERO)
- Precomputed LIBERO cache: [Chedan86/WorldPilot-LIBERO-precompute](https://huggingface.co/datasets/Chedan86/WorldPilot-LIBERO-precompute)

<a id="citation"></a>
## 📄 Citation

If WorldPilot helps your research, we would appreciate a citation using the BibTeX entry below.

```bibtex
@article{worldpilot2026,
  title={World Pilot: Steering Vision-Language-Action Models with World-Action Priors},
  author={Zefu Lin and Rongxu Cui and Junjia Xu and Xiaojuan Jin and Wenling Li and Lue Fan and Zhaoxiang Zhang},
  journal={Coming Soon.},
  year={2026}
}
```

<a id="acknowledgements"></a>
## 🤝 Acknowledgements

<p>
  We sincerely thank the teams behind <a href="https://github.com/amap-cvlab/ABot-Manipulation">ABot-Manipulation</a>,
  <a href="https://github.com/NVlabs/cosmos-policy">cosmos-policy</a>,
  <a href="https://github.com/Lifelong-Robot-Learning/LIBERO">LIBERO</a>,
  <a href="https://github.com/sylvestf/LIBERO-plus">LIBERO-plus</a>,
  <a href="https://github.com/huggingface/lerobot">LeRobot</a> for their outstanding work.
</p>
