import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

const repoRoot = process.env.QWEN3_ASR_MLX_RUNTIME_ROOT ?? process.cwd();
const bridge = spawn("scripts/qwen3-asr-mlx-bridge", [
  "Qwen/Qwen3-ASR-0.6B",
  "--local-files-only",
], { cwd: repoRoot });

const lines = createInterface({ input: bridge.stdout });
lines.on("line", line => {
  console.log(JSON.parse(line));
});

bridge.stderr.on("data", chunk => {
  process.stderr.write(chunk);
});

bridge.stdin.write(JSON.stringify({ type: "start" }) + "\n");
bridge.stdin.write(JSON.stringify({
  type: "transcribe",
  audio: process.argv[2] ?? "examples/audio/sample.wav",
  max_new_tokens: 0,
}) + "\n");
bridge.stdin.write(JSON.stringify({ type: "stop" }) + "\n");
