# Git Submodule Integration

Use this repository as a submodule when a parent app should pin an exact runtime
version but keep the ASR runtime independently releasable.

## Add To A Parent Repo

```bash
git submodule add https://github.com/YOUR_ORG/qwen3-asr-mlx-runtime.git Vendor/qwen3-asr-mlx-runtime
git submodule update --init --recursive
```

Commit the parent app changes:

```bash
git add .gitmodules Vendor/qwen3-asr-mlx-runtime
git commit -m "Add Qwen3-ASR MLX runtime submodule"
```

## Launch From Parent App

```bash
Vendor/qwen3-asr-mlx-runtime/scripts/qwen3-asr-mlx-bridge --local-files-only
```

## Pin A Runtime Update

```bash
cd Vendor/qwen3-asr-mlx-runtime
git fetch origin
git checkout <tag-or-commit>
cd ../..
git add Vendor/qwen3-asr-mlx-runtime
git commit -m "Update Qwen3-ASR MLX runtime"
```

## Recommended Parent Layout

```text
ParentApp/
  Vendor/
    qwen3-asr-mlx-runtime/   # submodule
  Sources/
  Tests/
```

Do not copy model weights into the parent app repo. Let the runtime download to
the configured model cache, or preinstall model snapshots in deployment.
