# WSG & WSV: World Snapshot Graphics

[[Standard]](https://world-snapshot.github.io/doc/index.html?page=S2_standard/01_overview.md) |
[[Reader]](https://github.com/World-Snapshot/wsg_reader) |
[[World Snapshot Models]](https://github.com/World-Snapshot/WSMs) |
[[WebUI]](https://github.com/World-Snapshot/ControlWorld) |
[[Documentation]](https://world-snapshot.github.io/doc/index.html) |
[[Organization]](https://github.com/World-Snapshot)

World Snapshot Graphics (`.wsg`) is a 256-channel image format that records a world snapshot on one two-dimensional image plane. RGB, geometry, motion, human pose, object structure, physical properties, materials, semantics, audio, and other machine-usable information remain spatially aligned inside the same file.

World Snapshot Video (`.wsv`) is the multi-frame form of WSG. Together, WSG and WSV provide the data boundary used by [World Snapshot Models](https://github.com/World-Snapshot/WSMs): RGB is one part of a snapshot, rather than the endpoint of Computer Vision.

This repository contains the complete **WSG/WSV Maker** and **WSG Reader / WSV Player**. If you only need to read or visualize existing files, use the smaller [wsg_reader](https://github.com/World-Snapshot/wsg_reader) repository.

## Format

| Format | Unit | Repository representation |
|---|---|---|
| `.wsg` | One world snapshot | NumPy-compatible archive containing an `H × W × 256` array, normally under `wsg_data` |
| `.wsv` | An ordered snapshot sequence | NumPy/ZIP-compatible archive containing frames such as `index_00000001`, `index_00000002`, ... |

The channel axis stores defined variables rather than arbitrary feature dimensions. Channels 0–2 store RGB; the remaining channels store dense values, sparse structures, identifiers, encoded payloads, or reserved data according to the [WSG standard](https://world-snapshot.github.io/doc/index.html?page=S2_standard/01_overview.md). The value `-1` denotes an unavailable or undefined location unless a channel contract specifies otherwise.

Raw WSG preserves injected values. Normalized WSG is used for model training and must carry reversible metadata: the modality definition, linear or logarithmic mode, and value range. `main/normalized_wsg_loader.py` uses a normalized file only when this information is available; otherwise it safely reads the raw WSG.

Non-RGB channels are not ordinary pseudo-RGB images. The reader applies each modality's official interpretation, palette, value range, and legend.

## Pipeline

WSG construction has three direct stages:

1. **Extract:** obtain world information from images, videos, meshes, annotations, or sensors.
2. **Inject:** write the extracted values and their spatial relationships into defined WSG channels.
3. **Read:** interpret, visualize, reconstruct, or export the requested information.

The scripts follow the same naming convention:

```text
pipelineNN_extract_*   source data → modality result
pipelineNN_inject_*    modality result → WSG/WSV channels
pipelineNN_read_*      WSG/WSV channels → values and official visualization
```

Single-frame WSG files can be merged into one WSV archive with `pipeline32_merge_wsg_to_wsv.py`. Large datasets can therefore be built frame by frame and deployed as compact sequential archives.

## Sources Used in WSM v1

The reported WSM v1 data was built with the following extraction tools and source datasets. This section lists what entered the released training/evaluation system, rather than every backend explored during development.

### Extraction Tools

1. [DWPose](https://github.com/IDEA-Research/DWPose): 2D articulated human structure.
2. [SMPLer-X](https://github.com/caizhongang/SMPLer-X): 3D human mesh and body parameters.
3. [MediaPipe Pose](https://github.com/google-ai-edge/mediapipe): 3D skeletal human representation.
4. [MediaPipe Seg](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter): human-part segmentation.
5. [PrimeDepth](https://github.com/vislearn/PrimeDepth): relative depth.
6. [Depth Pro](https://github.com/apple/ml-depth-pro): metric depth.
7. [NormalCrafter](https://binyr.github.io/NormalCrafter/): surface normals.
8. The repository's curvature and edge extraction pipelines.
9. [PTLFlow](https://github.com/hmorimitsu/ptlflow) with FlowNetSD: optical flow.

Object structure, physical properties, material values, PBR maps, semantic instances, and audio were injected from the source annotations or rendered assets below instead of being re-estimated by an additional model.

### Source Datasets

- [AVIS](https://github.com/ruohaoguo/avis): RGB/video, relative and metric depth, curvature, edge, optical flow, human representations, semantic instances, captions, and frame-aligned audio.
- [Articulation-XL 2.0](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0): object category, mesh, vertices, rig, UV, joints, and articulated structure.
- [ShapeNetSem](https://shapenet.org/): object structure, category, physical properties, conventional material attributes, and optical attributes.
- [MatSynth](https://gvecchio.com/matsynth/): base color, diffuse, normal, height, roughness, metallic, specular, opacity, and other PBR material maps.
- XPose: a separately prepared RGB/pose cache used for the released channels 122–124.

Almost all currently defined WSG information can be estimated from visual inputs. When reliable measurements or annotations already exist, injecting them directly avoids unnecessary re-estimation and preserves a cleaner training target.

## Explored or Retained References Not Used in WSM v1

The repository preserves several alternatives and prototypes that informed development but did **not** contribute to the WSM v1 results reported in the paper. Their presence in the repository does not mean that the released model was trained on them.

### Tools and Backends

- [Sapiens](https://github.com/facebookresearch/sapiens): retained extraction prototypes; the released human segmentation path uses MediaPipe Seg.
- [Depth Any Camera](https://github.com/yuliangguo/depth_any_camera) and [Metric3D v2](https://github.com/YvanYin/Metric3D): depth/normal alternatives considered before the final PrimeDepth, Depth Pro, and NormalCrafter paths.
- [DPFlow](https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/dpflow): considered as an optical-flow alternative; the released extraction script uses PTLFlow FlowNetSD.
- [VGGT](https://vgg-t.github.io/) and [MonST3R](https://monst3r-project.github.io/): extract-only experimental pipelines for camera and temporal geometry; no corresponding released WSG injection path was used in WSM v1.
- [ParticleSfM](https://github.com/bytedance/particle-sfm): camera and point-track reference.
- [Tokenize Anything](https://github.com/baaivision/tokenize-anything): text segmentation and visual-annotation reference.
- [Neural LightRig](https://projects.zxhezexin.com/neural-lightrig/): material/illumination reference.
- [SF3D](https://stable-fast-3d.github.io/) and [Pixel3DMM](https://simongiebenhain.github.io/pixel3dmm/): UV, geometry, normal, and material alternatives.

### Datasets

- [Meerkat](https://github.com/schowdhury671/meerkat): audio-visual reference considered as an AVIS alternative.
- [DynPose-100K](https://huggingface.co/datasets/nvidia/dynpose-100k), [Panda-70M](https://snap-research.github.io/Panda-70M/), and [Koala-36M](https://github.com/KwaiVGI/Koala-36M): camera, pose, video, and caption references.
- [UniRig](https://zjp-shadow.github.io/works/UniRig/): articulated-rig alternative.
- [DAM](https://huggingface.co/datasets/nvidia/describe-anything-dataset) and [RAM](https://huggingface.co/datasets/xinyu1205/recognize-anything-dataset): mask-description and recognition references.
- [Multiple Light Source Dataset](https://github.com/Visillect/mls-dataset): illumination reference.
- [ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore): considered alongside ShapeNetSem; the reported WSM v1 physical-property data uses ShapeNetSem.
- [MMFi](https://ntu-aiot-lab.github.io/mm-fi): WSG injection/reader support and a prepared cache exist, but the wireless channels were not included in the released WSM v1 task graph or reported evaluation.
- [UKBOB](https://emmanuelleb985.github.io/ukbob/): object/physical-property reference.
- [image2mass](https://github.com/tstandley/image2mass), [M3-VOS](https://huggingface.co/datasets/Lijiaxin0111/M3_VOS), and [ImageNet](https://image-net.org/): investigated as possible future mass, phase-transition, and general-image sources.

## Quick Start

Clone the full maker and reader:

```bash
git clone https://github.com/World-Snapshot/wsg.git
cd wsg/main
```

The extraction backends use separate environments because pose, depth, geometry, rendering, and sensor models have different dependencies. Environment definitions are stored in `main/env/`.

For the complete workflow, use:

```text
pipeline_one_stop_extract.ipynb
pipeline_one_stop_inject.ipynb
pipeline_one_stop_read.ipynb
pipeline_one_stop_tell_you_how_to_process_dataset.ipynb
```

The normal sequence is:

1. Place the source dataset under `input/`, or set its external path in the selected pipeline.
2. Run `pipeline_one_stop_extract.ipynb` to create the required environments and extract available modalities.
3. Run `pipeline_one_stop_inject.ipynb` to initialize RGB and inject the extracted information.
4. Run `pipeline_one_stop_read.ipynb` to verify values and official visualizations.
5. Merge validated frame files into `wsv/merge.wsv` when a sequential archive is required.

Each stage can also be run through its individual Python script. Dataset paths and model checkpoints remain explicit because different source datasets expose different contracts.

To read WSG/WSV without installing every extraction backend:

```bash
git clone https://github.com/World-Snapshot/wsg_reader.git
```

## AVIS Semantic and Audio Injection

The current AVIS cache uses one merged `wsv/merge.wsv` per video. Semantic and audio injection preserve the complete frame-level relationship:

- channel 195 stores every valid AVIS category inside its own instance mask;
- channels 175–184 store frame-aligned mixed audio over the union of all valid instance masks;
- simultaneous instances are retained instead of being reduced to the largest object;
- frame-to-frame audio amplitude is preserved;
- incomplete audio payloads are never written; and
- each WSV is atomically replaced only after a successful rewrite.

Check paths and annotations without modifying data:

```bash
cd main

python pipeline29_inject_avis_semantic_instance_into_wsv.py \
  --base-output-dir ../output \
  --subsets train val test \
  --dry-run

python pipeline18_inject_audio_into_wsg.py \
  --base-output-dir ../output \
  --subsets train val test \
  --dry-run
```

Remove `--dry-run` after the reported video, annotation, WAV, and WSV paths are correct. Both pipelines support deterministic sharding for large datasets.

## Reader and Visualization

The reader has three jobs:

- **Interpret:** recover the stored numerical, categorical, structural, or encoded value.
- **Visualize:** render it with the modality's official palette and global value definition.
- **Save:** export images, structured values, audio, meshes, or other modality-specific results.

Shared categorical palettes are stored in `main/configs/channel_visualization_palettes.json`. The readers under `main/pipeline*_read_*.py` are also used by WSM evaluation and ControlWorld, keeping data construction, evaluation, paper figures, and WebUI inference under the same visualization contract.

## Repository Structure

```text
wsg/
├── main/
│   ├── env/                         # Per-backend environments
│   ├── configs/                     # Visualization and channel configuration
│   ├── tools/                       # Dataset preprocessing tools
│   ├── pipeline00_*                 # Frame extraction and WSG initialization
│   ├── pipeline01-20_*              # Modality extract/inject/read pipelines
│   ├── pipeline29-33_*              # WSV injection, repair, merge, and release tools
│   └── pipeline_one_stop_*.ipynb    # Complete workflows
├── common/                          # Shared model and geometry utilities
├── data/                            # Dataset adapters
├── input/                           # Local input workspace
├── output/                          # Generated result workspace
├── pretrained_models/               # Local extraction checkpoints
├── LICENSE
└── README.md
```

## Integration

- [WSMs](https://github.com/World-Snapshot/WSMs) trains task routes on WSG variables and produces requested WSG targets.
- [ControlWorld](https://github.com/World-Snapshot/ControlWorld) selects a task, runs WSM inference, and displays the target through its WSG Reader.
- [wsg_reader](https://github.com/World-Snapshot/wsg_reader) provides the portable read/visualize subset without the extraction toolchain.

In a historical pose-domain measurement, 12 GB of source data produced 382 GB of intermediate visualizations and at least 100 GB of valid extracted information; injecting the machine-usable values into WSG increased final storage by 66 GB. WSG retains substantially more information than RGB without requiring every intermediate visualization to be distributed.

## License

The original WSG/WSV format implementation, registry, normalization,
serialization, reader/writer code, and documentation in this repository are
released under the [WSM Non-commercial Research License](LICENSE). Use is
limited to academic, research, experimental, personal, and other
noncommercial purposes. For commercial-use rights, contact Rutgers at
`compliance@research.rutgers.edu` and reference Rutgers Docket #2027-020.
Any Rutgers-owned model artifacts distributed with this repository, including
trained or fine-tuned weights, checkpoints, adapters, and `.pth` files, are
covered by the same license.

Third-party extraction tools, pretrained models, datasets, and adapted
components retain their respective licenses and terms. The RU-NCRL does not
relicense or claim ownership of those third-party materials.

---

WSG turns geometry, motion, pose, physics, materials, semantics, audio, and other information into addressable variables on the same image plane. Once information shares this boundary, it can be read, generated, transformed, and connected by one visual system.
