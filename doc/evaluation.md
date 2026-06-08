# Evaluation

Public evaluation in this repo targets `WorldPilot` on `Libero-Plus`.

Two entrypoints are provided:

- single-suite evaluation
- parallel batch evaluation across multiple suites

## 1. Prepare the environments

You need:

- `libero-plus` for the Libero-Plus simulator side
- `WorldPilot` for the policy server
- `cosmos-policy` available to serve Cosmos through the provided launcher

Install details are in [Installation](./environment.md).

## 2. Prepare files

Before evaluation, make sure you already have:

- a WorldPilot checkpoint
- a Cosmos checkpoint
- the Cosmos dataset statistics json
- the Cosmos T5 embedding pickle
- a valid `~/.libero-plus/config.yaml`

If you use the standard released Cosmos assets, the T5 embedding pickle from
[nvidia/Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)
is usually enough.

If you need to evaluate on `LIBERO-plus` language variants or extend the instruction cache manually,
run this in the `cosmos-policy` environment:

```bash
cd /path/to/WorldPilot

LIBERO_PLUS_ROOT=/path/to/LIBERO-plus \
LIBERO_CONFIG_PATH=~/.libero-plus \
mamba run -n cosmos-policy python -m cosmos_bridge.precompute_t5_embeddings \
  --pkl /path/to/libero_t5_embeddings.pkl
```

This script scans the `LIBERO-plus` task definitions and updates the T5 cache for those instructions.

## 3. Single-suite evaluation

Edit the top block of `examples/LIBERO-plus/eval_files/eval_libero_single.sh`.

The fields marked `Required` in that script are the only ones that must be filled before launch. Common knobs such as suite name, GPU, ports, start task id, and success count are in the same block.

Run:

```bash
cd /path/to/WorldPilot
mamba activate libero-plus
bash examples/LIBERO-plus/eval_files/eval_libero_single.sh
```

## 4. Batch evaluation

Edit the top block of `examples/LIBERO-plus/eval_files/eval_libero_batch.sh`.

Set:

- checkpoint and python paths
- Cosmos files
- Libero-Plus root and config path
- `SUITES`
- `GPUS`
- `COSMOS_PORTS`
- `POLICY_PORTS`

Run:

```bash
cd /path/to/WorldPilot
mamba activate libero-plus
bash examples/LIBERO-plus/eval_files/eval_libero_batch.sh
```

## 5. Behavior of the public eval scripts

The current scripts are aligned with the original internal flow in three important ways:

1. Cosmos is served online through `cosmos_bridge/run_cosmos_server.sh`
2. the policy is served through `deployment/model_server/server_policy.py`
3. simulator jobs talk to the policy server over sockets

The public version mainly changes how configuration is provided: instead of requiring a long chain of `export` commands, the required paths and GPU settings are collected at the top of each `.sh` file.
