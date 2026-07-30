from faster_whisper import WhisperModel
from pathlib import Path
import glob
import os
import site
import sys
import time

# =========================================
# CONFIGURACOES
# =========================================
# Pode ser audio ou video. Tambem da para passar o arquivo pelo terminal:
# python transcrever_offline.py "C:\caminho\arquivo.mp3"
DEFAULT_MEDIA_PATH = r"C:\Users\Lucas Anael\Downloads\WhatsApp Ptt 2026-07-30 at 14.55.12.ogg"
MODEL_PATH = r".\modelos\faster-whisper-large-v3"

LANGUAGE = "pt"
PREFER_GPU = True
GPU_COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
USE_VAD = True

CUDA_ERROR_FRAGMENTS = (
    "cuda",
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
)

SUPPORTED_MEDIA_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}

# =========================================
# AUXILIARES
# =========================================
def format_srt_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def resolve_media_path() -> Path:
    media_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_MEDIA_PATH)

    if media_path.exists() or media_path.suffix:
        return media_path

    matches = [
        media_path.with_suffix(extension)
        for extension in SUPPORTED_MEDIA_EXTENSIONS
        if media_path.with_suffix(extension).exists()
    ]

    if len(matches) == 1:
        return matches[0]

    return media_path

def validate_media_file(media_file: Path) -> None:
    if not media_file.exists():
        raise FileNotFoundError(f"Arquivo de audio/video nao encontrado: {media_file}")

    if not media_file.is_file():
        raise FileNotFoundError(f"O caminho informado nao e um arquivo: {media_file}")

    extension = media_file.suffix.lower()
    if extension and extension not in SUPPORTED_MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_EXTENSIONS))
        print(f"Aviso: extensao '{extension}' nao esta na lista comum: {supported}")
        print("Vou tentar transcrever mesmo assim.")

def candidate_site_packages():
    paths = []
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if user_site:
            paths.append(user_site)
    except Exception:
        pass

    for path in sys.path:
        if path and "site-packages" in path:
            paths.append(path)

    dedup = []
    seen = set()
    for path in paths:
        absolute_path = os.path.abspath(path)
        if absolute_path not in seen and os.path.isdir(absolute_path):
            seen.add(absolute_path)
            dedup.append(absolute_path)
    return dedup

def find_nvidia_dll_dirs():
    dll_dirs = []

    patterns = [
        os.path.join("nvidia", "cublas", "bin"),
        os.path.join("nvidia", "cudnn", "bin"),
        os.path.join("nvidia", "cuda_nvrtc", "bin"),
    ]

    for site_package in candidate_site_packages():
        for relative_path in patterns:
            full_path = os.path.join(site_package, relative_path)
            if os.path.isdir(full_path):
                dll_dirs.append(full_path)

        wildcard_bins = glob.glob(os.path.join(site_package, "nvidia", "*", "bin"))
        for path in wildcard_bins:
            if os.path.isdir(path):
                dll_dirs.append(path)

    final = []
    seen = set()
    for dll_dir in dll_dirs:
        absolute_path = os.path.abspath(dll_dir)
        if absolute_path not in seen:
            seen.add(absolute_path)
            final.append(absolute_path)
    return final

def inject_nvidia_dlls():
    dll_dirs = find_nvidia_dll_dirs()

    if not dll_dirs:
        print("Nenhuma pasta de DLL da NVIDIA encontrada no site-packages.")
        return []

    print("Pastas de DLL detectadas:")
    for dll_dir in dll_dirs:
        print(" -", dll_dir)

    for dll_dir in dll_dirs:
        try:
            os.add_dll_directory(dll_dir)
        except (AttributeError, FileNotFoundError, OSError):
            pass

    os.environ["PATH"] = os.pathsep.join(dll_dirs + [os.environ.get("PATH", "")])

    return dll_dirs

def load_cpu_model(model_path: str):
    model = WhisperModel(
        model_path,
        device="cpu",
        compute_type=CPU_COMPUTE_TYPE,
    )
    print("CPU ativada.")
    return model, "cpu", CPU_COMPUTE_TYPE

def load_model_with_fallback(model_path: str):
    if PREFER_GPU:
        try:
            print("Tentando carregar com GPU...")
            inject_nvidia_dlls()
            model = WhisperModel(
                model_path,
                device="cuda",
                compute_type=GPU_COMPUTE_TYPE,
            )
            print("GPU ativada com sucesso.")
            return model, "cuda", GPU_COMPUTE_TYPE
        except Exception as error:
            print(f"Falha ao iniciar GPU: {error}")
            print("Voltando para CPU...")

    return load_cpu_model(model_path)

def is_cuda_runtime_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(fragment in message for fragment in CUDA_ERROR_FRAGMENTS)

def transcribe_segments(model: WhisperModel, media_file: Path):
    segments, info = model.transcribe(
        str(media_file),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=USE_VAD,
    )
    return list(segments), info

def transcribe_with_runtime_fallback(model: WhisperModel, model_dir: Path, media_file: Path, device_used: str):
    try:
        segments, info = transcribe_segments(model, media_file)
        return segments, info, device_used, GPU_COMPUTE_TYPE if device_used == "cuda" else CPU_COMPUTE_TYPE
    except RuntimeError as error:
        if device_used != "cuda" or not is_cuda_runtime_error(error):
            raise

        print(f"Falha durante a transcricao com GPU: {error}")
        print("A GPU carregou, mas faltou alguma DLL CUDA/cuBLAS/cuDNN.")
        print("Refazendo a transcricao com CPU...")
        cpu_model, cpu_device, cpu_compute_type = load_cpu_model(str(model_dir))
        segments, info = transcribe_segments(cpu_model, media_file)
        return segments, info, cpu_device, cpu_compute_type

def write_txt(output_txt: Path, segments) -> None:
    print("Gerando TXT...")
    with open(output_txt, "w", encoding="utf-8") as file:
        for segment in segments:
            text = segment.text.strip()
            if text:
                file.write(text + "\n")

def write_srt(output_srt: Path, segments) -> None:
    print("Gerando SRT...")
    with open(output_srt, "w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            text = segment.text.strip()
            if text:
                file.write(f"{index}\n")
                file.write(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n")
                file.write(f"{text}\n\n")

def main() -> None:
    media_file = resolve_media_path()
    model_dir = Path(MODEL_PATH)

    validate_media_file(media_file)

    if not model_dir.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_dir}")

    output_txt = media_file.with_suffix(".txt")
    output_srt = media_file.with_suffix(".srt")

    start_time = time.time()

    print("Arquivo:", media_file)
    print("Carregando modelo local...")
    model, device_used, compute_type_used = load_model_with_fallback(str(model_dir))

    print("Iniciando transcricao...")
    segments, info, device_used, compute_type_used = transcribe_with_runtime_fallback(
        model,
        model_dir,
        media_file,
        device_used,
    )

    write_txt(output_txt, segments)
    write_srt(output_srt, segments)

    end_time = time.time()

    print("\nConcluido com sucesso.")
    print("Dispositivo usado:", device_used)
    print("Compute type usado:", compute_type_used)
    print("Idioma detectado:", info.language)
    print("Probabilidade:", info.language_probability)
    print("TXT:", output_txt)
    print("SRT:", output_srt)
    print(f"Tempo total: {(end_time - start_time) / 60:.2f} minutos")

if __name__ == "__main__":
    main()
