# PostFuse
Unimodal SER baselines under a shared log-Mel protocol, plus TS-α posterior fusion.
We adapt eleven representative networks (CNN, ResNet, DenseNet, CLDNN, RNN, LSTM, BiLSTM, GRU, GCN, ViT, VLP-ViT) to matched spectrogram or frame-sequence inputs. PostFuse combines independently trained members using class posteriors only. TS-α jointly fits temperatures, simplex weights, and Amari α.
## Layout
- `models/` — SER-adapted member networks and log-Mel packing
## License
Code in this repository is released under the MIT License (see `LICENSE`), except third-party material listed below.

## Data

This repository does not include audio. Experiments use two public acted SER corpora:

- **CASIA** Chinese emotional speech (Institute of Automation, CAS). Six classes in the main tables.
- **EmoDB** Berlin emotional speech (Burkhardt et al.). Used as a same-protocol second-corpus check.

Obtain each corpus from its official distributor and follow the original license/terms. Place the files according to the paths expected by your own training script. We do not redistribute either corpus.

### Third-party code
**VLP-ViT (adapted)**  
Source: https://github.com/kjy7567/speech_emotion_recognition_from_log_Mel_spectrogram_using_vertically_long_patch  
Paper: Kim et al., IEEE Access, 2024. https://doi.org/10.1109/ACCESS.2024.3447770  
The upstream repository does not declare a license. We release only our SER adaptation (student ViT, 128×128 log-Mel) and do not redistribute the upstream checkout.
Classic CNN/ResNet/DenseNet-style members are our reimplementations for this protocol; cite the original architecture papers in the manuscript.
