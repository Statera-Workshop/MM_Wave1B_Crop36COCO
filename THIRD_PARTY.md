# Third-party code and data

| Component | Source | Licence | Used for |
| --- | --- | --- | --- |
| BPE tokenizer `simple_tokenizer.py` and `bpe_simple_vocab_16e6.txt.gz` | [openai/CLIP](https://github.com/openai/CLIP) | MIT (`licenses/openai_CLIP_LICENSE.txt`) | vendored unchanged in `code/third_party/` |
| CLIP and SigLIP loss definitions | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) `loss.py` | MIT (`licenses/open_clip_LICENSE.txt`) | `contrastive_loss` in `code/mm_b.py` (re-implemented, single GPU) |
| VSE++ max-violation hinge (and the VSE0 sum-of-hinges baseline) | [fartashf/vsepp](https://github.com/fartashf/vsepp) | Apache-2.0 (`licenses/vsepp_LICENSE.txt`) | `contrastive_loss` in `code/mm_b.py` (re-implemented) |

**Data (not redistributed).** COCO 2017 captions (CC BY 4.0); COCO images under the Flickr terms of use - only the
image ids of the subset are published. Crop36 / FashionLetters36 are data of this series.
