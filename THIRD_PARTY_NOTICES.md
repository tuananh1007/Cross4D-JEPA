# Third-party notices

## P4Transformer

`src/cross4d_jepa/models/encoders/p4t_official/` vendors modules from the official
P4Transformer implementation (https://github.com/hehefan/P4Transformer,
`modules-pytorch-1.8.1/`), with only the internal imports made relative. It is
distributed under the following license:

```
MIT License

Copyright (c) 2021 Hehe Fan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`pointnet2_utils.py` and `pytorch_utils.py` in that directory derive from
Pointnet2_PyTorch (https://github.com/erikwijmans/Pointnet2_PyTorch) and keep their
original MIT copyright headers (Facebook, Inc. and its affiliates).

## Pretrained teachers and datasets

Teacher weights (DINOv2, DINOv3, V-JEPA 2, VideoMAE V2, ViT-B/16) are downloaded from
their original sources at run time and are **not** redistributed here; they remain
under their own licenses. The same holds for every dataset (MSR-Action3D, NTU RGB+D
60, DeformingThings4D, HOI4D): obtain each from its original source under its own
terms.
