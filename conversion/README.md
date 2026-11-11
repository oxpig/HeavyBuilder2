# Converting the weights to float32

HeavyBuilder2's weights are published on Zenodo
(https://doi.org/10.5281/zenodo.24115156) as float32.  This directory
records how they were converted from the float64 files first distributed
through Google Drive:

- `convert_to_float32.py`: the conversion script.

## Why

The three weight files were PyTorch state dicts saved as float64, 490 MB
in all.  HeavyBuilder2 builds its models in float32, so `load_state_dict`
casts the weights to float32 when loading them, and the extra precision
was never used.  Stored as float32, the files take 245 MB.

## How

`convert_to_float32.py` loads each state dict, converts every float64
tensor to float32, and refuses to write a file unless every value
converts back to exactly the original.  It then reloads what it wrote
and checks it again.

```
python convert_to_float32.py <directory of float64 files> <output directory>
```

## Checks

- All 61.2 million values, in all three files, are exactly representable
  in float32, so the conversion loses nothing.
- For trastuzumab's heavy chain, HeavyBuilder2's unrefined models and
  error estimates were byte-identical with the float64 and float32
  files.
- The refined models differed by a Cα RMSD of 0.03–0.05 Å.  That is the
  run-to-run variation of the OpenMM refinement itself: two runs with the
  float64 files differ by 0.035 Å.

## Checksums

SHA-256 of the original float64 files:

```
89d11cb31afcf48a5698d8f639a9275c7b890e52b83df23f7f330784d6a5325f  heavy_chain_model_1
3cc1edabf12c8175bfbe49357d087f50e252e7102cc70bc91442bbb5692c4b1b  heavy_chain_model_2
9a09aad0ac7a9cfa0d95724c2503f82bbc1e91bd94428eab47eeedf124b6db13  heavy_chain_model_3
```

SHA-256 of the float32 files on Zenodo, as `HeavyBuilder/weights.py`
checks them:

```
ca8d974f0c43ad80463847898d7cc31ee3cc3d4ca31854ce2b6f722321888a77  heavy_chain_model_1
a85496b331f4249ae24b91177a51c510be207f2ec80b55594dfd43d5bff64c9e  heavy_chain_model_2
98164168764f53c81b499574afdcd40f09d90a77aab07bcd9d46e8c04f5ee6d2  heavy_chain_model_3
```

A fourth model, `heavy_chain_model_4`, was never used by HeavyBuilder2
and was neither converted nor published.
