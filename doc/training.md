# Training

The public training path is:

1. prepare the `WorldPilot` and `cosmos-policy` environments
2. prepare the required pretrained weights and cache artifacts
3. precompute or download the Cosmos cache for Libero
4. edit the Libero training yaml and launch script
5. run training

## 0. Required pretrained assets

The current training config is [`examples/LIBERO/train_files/WorldPilot.yaml`](../examples/LIBERO/train_files/WorldPilot.yaml).

- Cosmos Policy LIBERO checkpoint: [nvidia/Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)
- VGGT: [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)
- Qwen3-VL action checkpoint: [StarVLA/Qwen3-VL-4B-Instruct-Action](https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action)
- ABot-M0 pretrain checkpoint: [amap_cvlab/ABot-M0-Pretrain](https://www.modelscope.cn/models/amap_cvlab/ABot-M0-Pretrain)

## 1. Download or prepare Cosmos cache

Training reads precomputed Cosmos cache from `datasets.vla_data.cosmos_cache_dir`.

For standard LIBERO training, you do not need to run `cosmos_bridge.precompute_t5_embeddings.py`.
The released `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` package already includes a usable
`libero_t5_embeddings.pkl`, and that file is sufficient for Cosmos cache generation on the LIBERO task suites
used by this repo.

If you want to reuse a published cache, download it from Hugging Face:

- [Chedan86/WorldPilot-LIBERO-precompute](https://huggingface.co/datasets/Chedan86/WorldPilot-LIBERO-precompute)

If you do not have a published cache yet, generate it locally with the steps below. The directory layout is
expected to match the dataset split names under `cosmos_cache_dir`, for example:

```text
/path/to/cosmos_cache/
  libero_10_no_noops_1.0.0_lerobot/
  libero_goal_no_noops_1.0.0_lerobot/
  libero_object_no_noops_1.0.0_lerobot/
  libero_spatial_no_noops_1.0.0_lerobot/
```

## 2. Precompute Cosmos cache

Edit the variables required by `cosmos_bridge/run_precompute.sh`, then run:

```bash
cd /path/to/WorldPilot
bash cosmos_bridge/run_precompute.sh
```

The script starts one Cosmos server per dataset split and writes the cache to your target output directory.

## 3. Edit the training config

Update `examples/LIBERO/train_files/WorldPilot.yaml` for your machine and experiment.

At minimum, set:

- `run_root_dir`
- `run_id`
- `framework.vggt_path`
- `framework.qwenvl.base_vlm`
- `datasets.vla_data.data_root_dir`
- `datasets.vla_data.cosmos_cache_dir`
- `trainer.pretrained_checkpoint`

## 4. Edit the launch script

Update the top block of `examples/LIBERO/train_files/run_libero_train.sh`:

- `GPU_IDS`
- `CONFIG_YAML` if you do not want the default yaml


## 5. Launch

```bash
cd /path/to/WorldPilot
mamba activate WorldPilot
bash examples/LIBERO/train_files/run_libero_train.sh
```

Outputs are written under the `run_root_dir/run_id` directory from the yaml.
