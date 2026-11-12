# DeepLense Test Tasks 2026
ML4SCI DeepLense evaluation tests. Common Test I is done here.

## Common Test I: 3-class classification
ResNet18 (pretrained, 1-channel input) classifying no / sphere / vort lensing images.

**Val accuracy 96.48% · macro AUC 0.9953**

![ROC + confusion](common_test_1/roc_confusion.png)

## Approach
| Item | What I did |
|---|---|
| Task | 3-class classification (no / sphere / vort), 150x150 single-channel images |
| Split | Given split: 30,000 train / 7,500 val (2,500 per class). The test doc mentions 90:10, but I kept the provided folders |
| Model | ResNet18 pretrained on ImageNet, `conv1` replaced with a 1-channel conv, `fc` replaced with a 3-class layer |
| Normalisation | Mean/std (0.0617 / 0.1171) from every 50th training image, not ImageNet values, since these are not natural images |
| Augmentation | Flips + rotation (20°). Lens images have no fixed orientation, so this should be safe |
| Training | Adam lr 1e-4, wd 1e-4, cosine decay to 1e-6, 15 epochs, batch 32, Resize 256 |
| Validation | Val was used for monitoring and for the final numbers. No separate test set, and earlier experiments also looked at val |
| Platform | Google Colab (T4). Dataset zip is on my Drive |

## Results (validation set)
| Metric | Value |
|---|---|
| Accuracy | 96.48% |
| Macro AUC | 0.9953 |
| AUC: no / sphere / vort | 0.9954 / 0.9922 / 0.9983 |

| Class | Precision | Recall |
|---|---|---|
| no | 0.94 | 0.99 |
| sphere | 0.97 | 0.93 |
| vort | 0.98 | 0.97 |

Sphere is the weakest class: 123 sphere images were called "no" and 51 "vort". The weights reload to the same accuracy (96.48%).

## Earlier runs
| Run | Result | Note |
|---|---|---|
| ResNet18, no norm, lr 1e-3 | ~92-94% | overfit, which is why I changed the recipe |
| ResNet18, lr 1e-4, manual LR drops, no scheduler | 96.28% | I picked the drops by watching val |
| Same cosine recipe, two earlier runs | 96.28% (AUC 0.9948), 96.43% (AUC 0.9951) | same code, different runs |

These come from earlier notebooks that are not in this repo, so treat them as self-reported.

## Limitations
- Val was used for monitoring and for the final numbers, with no separate test set.
- Seeded but not fully deterministic, so results vary slightly between runs. Three runs of the same recipe gave 96.28%, 96.43% and 96.48%.
- Single recipe, so I can't say which change (normalisation, lr, scheduler, resize) helped most.

## Files
- `common_test_1/common_test_1.ipynb`: final notebook
- `common_test_1/resnet18_common_test1.pth`: trained weights
- `common_test_1/roc_confusion.png`: ROC and confusion matrix