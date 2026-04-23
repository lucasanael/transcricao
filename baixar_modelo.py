from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_NAME = "large-v3"

destino = Path("modelos") / f"faster-whisper-{MODEL_NAME}"
destino.mkdir(parents=True, exist_ok=True)

print(f"Baixando modelo para: {destino.resolve()}")

snapshot_download(
    repo_id=f"Systran/faster-whisper-{MODEL_NAME}",
    local_dir=str(destino),
    local_dir_use_symlinks=False,
)

print("Download concluído com sucesso.")
print(destino.resolve())
