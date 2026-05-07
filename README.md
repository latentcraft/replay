
<h3 align="center"><a href="https://arxiv.org/abs/2511.19972" style="color:#9C276A">
Boosting Reasoning in Large Multimodal Models via Activation Replay</a></h3>

<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2511.19972-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.19972)<br>

## 🎉 News

- [2025/11/27] Arxiv is released.

## 🕹️ Approach
-  we propose **Activation Replay**, a simple yet effective training-free solution that boosts multimodal reasoning of post-trained LMMs, without requiring expensive policy optimization. By modulation of visual tokens at test time, Activation Replay enforces RLVR low-entropy activations to mimick the distributions from paired base LMMs at test time.
-  We validate our approach across diverse LMMs post-trained by RLVR and diverse reasoning scenarios, including **mathmatics**, **multi-turn agents** that perform visual search in high-resolution images, and **video reasoners** that think across frames, where our approach showcases consistent performance gains across these setups. 
<div align="center">
  <img src="assets/xy_repl_fig5.png" alt="Your Image" width="65%" style="float: left; margin-right: 1px;"/>
</div>

## 🔥 Performance

<div align="center">
  <img src="assets/xy_main_results.PNG" alt="Your Image" width="65%" style="float: left; margin-right: 1px;"/>
</div>

## ✒️ Citation
```
@article{xing2025boosting,
  title={Boosting Reasoning in Large Multimodal Models via Activation Replay},
  author={Xing, Yun and Hu, Xiaobin and He, Qingdong and Zhang, Jiangning and Yan, Shuicheng and Lu, Shijian and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2511.19972},
  year={2025}
}
```

## ❤️ Acknowledgement
Thanks for their wonderful work!
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git)
- [DeepEyes](https://github.com/Visual-Agent/DeepEyes)
- [Video-R1](https://github.com/tulerfeng/Video-R1)
