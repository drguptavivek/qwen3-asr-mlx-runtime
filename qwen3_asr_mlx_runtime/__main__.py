"""Module entrypoint for `python -m qwen3_asr_mlx_runtime`."""

from .bridge import main


if __name__ == "__main__":
    raise SystemExit(main())
