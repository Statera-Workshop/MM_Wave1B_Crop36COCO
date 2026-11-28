<!-- PAI-SG-LINKS:BEGIN -->
## Project Links

- [Project Portal](https://statera-workshop.github.io/#repo-mm_wave1b_crop36coco)
- [Statera Workshop Portal](https://statera-workshop.github.io/)
<!-- PAI-SG-LINKS:END -->

# MM_Wave1B_Crop36COCO

**Image-text contrastive convergence on Crop36-Text and COCO - PAI-SG Multimodal Wave 1B**

Part of the *Physical AI* reference series of the **Statera Working Group**. Convergence of image-text contrastive losses trained from scratch on Crop36-Text (generated captions for FashionLetters36 object crops) and a 20,000-image COCO subset. Method repositories: [MM_Contrastive_VSE](https://github.com/Statera-Workshop/MM_Contrastive_VSE), [MM_Contrastive_CLIP](https://github.com/Statera-Workshop/MM_Contrastive_CLIP), [MM_Contrastive_SigLIP](https://github.com/Statera-Workshop/MM_Contrastive_SigLIP).

## Protocol

**Common setup** (identical for the contrastive repositories; full comparison in [MM_Wave1B_Crop36COCO](https://github.com/Statera-Workshop/MM_Wave1B_Crop36COCO)).
Dual encoder from scratch: image ResNet-18 (torchvision layout), text Transformer (4 layers, width 256, 4 heads, causal,
EOT pooling; OpenAI CLIP BPE tokenizer), both projected to a 256-d L2-normalised embedding. AdamW lr 5e-4, wd 0.1,
500 warm-up steps then constant (cosine = ablation), batch 256, no augmentation, fp32, seeds 777 / 778 / 779.
Data: **Crop36-Text** - the 26,716 / 6,699 Crop36 object crops (96 x 96, grayscale, 36 classes) of FashionLetters36 with
captions from the class name and the original box size (8 templates); **COCO subset** - 20,000 train2017 / 1,000
val2017 images (seed 777, 128 x 128), 5 captions each. Metrics: Crop36 zero-shot top-1 (3 size-free prompts, ensemble;
chance 0.028); COCO rsum = Recall@1+5+10 image->text + text->image on the 1,000 val images (max 6).

**BatchNorm finding.** With a constant learning rate the BatchNorm running statistics lag the moving weights, so the
plain eval-mode score shows isolated *dips* (a single evaluation falls, the next recovers) and, on COCO, a systematic
under-estimate. Re-estimating the BN statistics on 50 training batches before evaluation (the series' evaluation
practice) removes every dip and gives the *BN re-estimated* column; training is unchanged. With the cosine schedule the
two evaluations agree. *Dips* = evaluation-to-evaluation drops >= 0.1 (>= 0.6 rsum).

## Results (mean of seeds; range in brackets)

### Crop36-Text

| Loss | Init | Batch | Epochs | LR schedule | Seeds | Final zero-shot top-1 (range) | Final, BN re-estimated (range) | E90 | Dips (plain / BN re-est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VSE0 (sum of hinges) | scratch | 256 | 30 | constant | 3 | 0.898 (0.026) | 0.907 (0.008) | 4.0 | 1 / - |
| VSE0 (sum of hinges) | scratch | 256 | 100 | constant | 3 | 0.894 (0.017) | 0.903 (0.001) | 7.0 | 0 / 0 |
| VSE0 (sum of hinges) | scratch | 64 | 30 | constant | 2 | 0.903 (0.017) | 0.910 (0.004) | 3.0 | 0 / - |
| VSE0 (sum of hinges) | SSL Barlow Twins | 256 | 30 | constant | 3 | 0.924 (0.005) | 0.924 (0.004) | 2.0 | 0 / 0 |
| VSE0 (sum of hinges) | SSL Barlow Twins | 256 | 100 | constant | 3 | 0.923 (0.005) | 0.923 (0.005) | 3.0 | 0 / 0 |
| VSE++ (max violation) | scratch | 256 | 30 | constant | 3 | 0.026 (0.005) | 0.026 (0.008) | 1.0 | 0 / - |
| VSE++ (max violation) | scratch | 256 | 100 | constant | 3 | 0.026 (0.004) | 0.029 (0.006) | 3.0 | 0 / - |
| VSE++ (max violation) | scratch | 64 | 30 | constant | 2 | 0.028 (0.006) | 0.026 (0.003) | 1.0 | 0 / - |
| CLIP | scratch | 256 | 30 | constant | 3 | 0.892 (0.024) | 0.903 (0.011) | 2.0 | 0 / - |
| CLIP | scratch | 256 | 100 | constant | 3 | 0.851 (0.134) | 0.899 (0.003) | 3.0 | 5 / 0 |
| CLIP | scratch | 256 | 100 | cosine | 3 | 0.910 (0.006) | 0.911 (0.006) | 3.0 | 2 / 0 |
| CLIP | scratch | 64 | 30 | constant | 3 | 0.908 (0.006) | 0.910 (0.009) | 2.7 | 0 / - |
| CLIP | SSL Barlow Twins | 256 | 30 | constant | 3 | 0.924 (0.006) | 0.924 (0.006) | 1.0 | 0 / 0 |
| CLIP | SSL Barlow Twins | 256 | 100 | constant | 3 | 0.917 (0.012) | 0.922 (0.003) | 3.0 | 1 / 0 |
| SigLIP | scratch | 256 | 30 | constant | 3 | 0.904 (0.007) | 0.910 (0.004) | 2.0 | 1 / - |
| SigLIP | scratch | 256 | 100 | constant | 3 | 0.902 (0.007) | 0.908 (0.003) | 3.0 | 11 / 0 |
| SigLIP | scratch | 256 | 100 | cosine | 3 | 0.919 (0.004) | 0.918 (0.002) | 3.0 | 0 / 0 |
| SigLIP | scratch | 64 | 30 | constant | 3 | 0.910 (0.001) | 0.915 (0.004) | 2.0 | 1 / - |
| SigLIP | SSL Barlow Twins | 256 | 30 | constant | 3 | 0.925 (0.001) | 0.925 (0.002) | 2.0 | 0 / 0 |
| SigLIP | SSL Barlow Twins | 256 | 100 | constant | 3 | 0.911 (0.041) | 0.928 (0.003) | 3.0 | 0 / 0 |

### COCO subset

| Loss | Init | Batch | Epochs | LR schedule | Seeds | Final rsum (R@1+5+10, both directions, max 6) (range) | Final, BN re-estimated (range) | E90 | Dips (plain / BN re-est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VSE0 (sum of hinges) | scratch | 256 | 15 | constant | 3 | 0.537 (0.116) | 0.594 (0.088) | 11.0 | 0 / - |
| VSE0 (sum of hinges) | scratch | 256 | 50 | constant | 3 | 0.730 (0.057) | 0.775 (0.064) | 26.7 | 0 / 0 |
| VSE0 (sum of hinges) | scratch | 256 | 100 | constant | 3 | 0.708 (0.047) | 0.744 (0.092) | 27.0 | 0 / 0 |
| VSE++ (max violation) | scratch | 256 | 15 | constant | 3 | 0.030 (0.009) | 0.031 (0.017) | 1.0 | 0 / - |
| VSE++ (max violation) | scratch | 256 | 50 | constant | 3 | 0.035 (0.017) | 0.039 (0.006) | 1.3 | 0 / - |
| VSE++ (max violation) | scratch | 256 | 100 | constant | 3 | 0.029 (0.008) | 0.036 (0.007) | 3.0 | 0 / - |
| CLIP | scratch | 256 | 15 | constant | 3 | 0.617 (0.059) | 0.673 (0.039) | 6.7 | 0 / - |
| CLIP | scratch | 256 | 50 | constant | 3 | 0.700 (0.031) | 0.733 (0.043) | 8.7 | 0 / 0 |
| CLIP | scratch | 256 | 100 | constant | 3 | 0.701 (0.074) | 0.740 (0.086) | 13.0 | 0 / 0 |
| CLIP | scratch | 256 | 100 | cosine | 3 | 0.771 (0.017) | 0.770 (0.011) | 27.0 | 0 / 0 |
| SigLIP | scratch | 256 | 15 | constant | 3 | 0.682 (0.119) | 0.763 (0.097) | 8.0 | 0 / - |
| SigLIP | scratch | 256 | 50 | constant | 3 | 0.678 (0.028) | 0.746 (0.010) | 8.0 | 0 / 0 |
| SigLIP | scratch | 256 | 100 | constant | 3 | 0.732 (0.069) | 0.751 (0.040) | 12.0 | 0 / 0 |
| SigLIP | scratch | 256 | 100 | cosine | 3 | 0.747 (0.039) | 0.746 (0.047) | 14.0 | 0 / 0 |

**SSL initialisation (Crop36).** Image encoder initialised from the SSL Barlow Twins 200-epoch backbone of [SSL_Redundancy_BarlowTwins](https://github.com/Statera-Workshop/SSL_Redundancy_BarlowTwins) (seed-matched; conv1 folded to one input channel), Barlow vs scratch: VSE0 30 epochs: E90 2.0 vs 4.0, final (BN re-est.) 0.924 vs 0.907; VSE0 100 epochs: E90 3.0 vs 7.0, final (BN re-est.) 0.923 vs 0.903; CLIP 30 epochs: E90 1.0 vs 2.0, final (BN re-est.) 0.924 vs 0.903; CLIP 100 epochs: E90 3.0 vs 3.0, final (BN re-est.) 0.922 vs 0.899; SigLIP 30 epochs: E90 2.0 vs 2.0, final (BN re-est.) 0.925 vs 0.910; SigLIP 100 epochs: E90 3.0 vs 3.0, final (BN re-est.) 0.928 vs 0.908.

## Findings

- **VSE++ (hardest negative only) collapses from scratch in every setting; VSE0 converges.**
- **Crop36: the three working losses reach the same level** after BN re-estimation (SigLIP marginally ahead); CLIP / SigLIP reach 90 % of the final value in 2-3 epochs, VSE0 in 4-7.
- **COCO: SigLIP is clearly fastest early (15 epochs); at 50-100 epochs the losses are within the seed range.** Under the constant schedule COCO stops improving after about 50 epochs.
- **Constant learning rate + BatchNorm distorts last-epoch reporting**: isolated dips and a COCO under-estimate, both removed by BN re-estimation; the cosine schedule raises the level and removes the need for it.

Full tables: `results/summary_B.md`; BN re-estimation of every saved model: `results/bn_reestimation_B.md`; COCO subset ids and their md5: `results/data/coco_sub_list.json`.

**Data note.** COCO captions CC BY 4.0, images under the Flickr terms (not redistributed). Crop36 is derived from FashionLetters36 (series data).

## Run

```bash
cd code && python add_crop36_side.py && python prep_coco.py
bash smoke_b.sh && python make_queue_b.py --print-only
```
