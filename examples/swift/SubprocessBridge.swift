import Foundation

let repoRoot = ProcessInfo.processInfo.environment["QWEN3_ASR_MLX_RUNTIME_ROOT"]
    ?? FileManager.default.currentDirectoryPath
let audioPath = CommandLine.arguments.dropFirst().first ?? "examples/audio/sample.wav"

let process = Process()
process.executableURL = URL(fileURLWithPath: "/bin/bash")
process.arguments = [
    "\(repoRoot)/scripts/qwen3-asr-mlx-bridge",
    "Qwen/Qwen3-ASR-0.6B",
    "--local-files-only"
]

let stdin = Pipe()
let stdout = Pipe()
let stderr = Pipe()
process.standardInput = stdin
process.standardOutput = stdout
process.standardError = stderr

try process.run()

let messages: [[String: Any]] = [
    ["type": "start"],
    ["type": "transcribe", "audio": audioPath, "max_new_tokens": 0],
    ["type": "stop"]
]

for message in messages {
    let data = try JSONSerialization.data(withJSONObject: message)
    stdin.fileHandleForWriting.write(data)
    stdin.fileHandleForWriting.write(Data("\n".utf8))
}
stdin.fileHandleForWriting.closeFile()

let output = stdout.fileHandleForReading.readDataToEndOfFile()
if let text = String(data: output, encoding: .utf8) {
    print(text)
}

let diagnostics = stderr.fileHandleForReading.readDataToEndOfFile()
if !diagnostics.isEmpty, let text = String(data: diagnostics, encoding: .utf8) {
    FileHandle.standardError.write(Data(text.utf8))
}

process.waitUntilExit()
