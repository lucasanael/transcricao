from faster_whisper import WhisperModel
from pathlib import Path
import os
import sys
import time
import site
import glob

# =========================================
# CONFIGURAÇÕES
# =========================================
VIDEO_PATH = r"C:\Users\Lucas Anael\Desktop\transcricao\reuniao.mp4"
MODEL_PATH = r".\modelos\faster-whisper-large-v3"

LANGUAGE = "pt"
PREFER_GPU = True
GPU_COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
USE_VAD = True

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

    for p in sys.path:
        if p and "site-packages" in p:
            paths.append(p)

    # remove duplicados preservando ordem
    dedup = []
    seen = set()
    for p in paths:
        p = os.path.abspath(p)
        if p not in seen and os.path.isdir(p):
            seen.add(p)
            dedup.append(p)
    return dedup

def find_nvidia_dll_dirs():
    dll_dirs = []

    patterns = [
        os.path.join("nvidia", "cublas", "bin"),
        os.path.join("nvidia", "cudnn", "bin"),
        os.path.join("nvidia", "cuda_nvrtc", "bin"),
    ]

    for sp in candidate_site_packages():
        for rel in patterns:
            full = os.path.join(sp, rel)
            if os.path.isdir(full):
                dll_dirs.append(full)

        # fallback: procura qualquer pasta bin dentro de nvidia
        wildcard_bins = glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
        for path in wildcard_bins:
            if os.path.isdir(path):
                dll_dirs.append(path)

    # remove duplicados
    final = []
    seen = set()
    for d in dll_dirs:
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            final.append(d)
    return final

def inject_nvidia_dlls():
    dll_dirs = find_nvidia_dll_dirs()

    if not dll_dirs:
        print("Nenhuma pasta de DLL da NVIDIA encontrada no site-packages.")
        return []

    print("Pastas de DLL detectadas:")
    for d in dll_dirs:
        print(" -", d)

    for d in dll_dirs:
        try:
            os.add_dll_directory(d)
        except (AttributeError, FileNotFoundError, OSError):
            pass

    # Também adiciona ao PATH do processo atual
    os.environ["PATH"] = os.pathsep.join(dll_dirs + [os.environ.get("PATH", "")])

    return dll_dirs

def load_model_with_fallback(model_path: str):
    if PREFER_GPU:
        try:
            print("Tentando carregar com GPU...")
            inject_nvidia_dlls()
            model = WhisperModel(
                model_path,
                device="cuda",
                compute_type=GPU_COMPUTE_TYPE
            )
            print("GPU ativada com sucesso.")
            return model, "cuda", GPU_COMPUTE_TYPE
        except Exception as e:
            print(f"Falha ao iniciar GPU: {e}")
            print("Voltando para CPU...")

    model = WhisperModel(
        model_path,
        device="cpu",
        compute_type=CPU_COMPUTE_TYPE
    )
    print("CPU ativada.")
    return model, "cpu", CPU_COMPUTE_TYPE

# =========================================
# VALIDAÇÕES
# =========================================
video_file = Path(VIDEO_PATH)
model_dir = Path(MODEL_PATH)

if not video_file.exists():
    raise FileNotFoundError(f"Vídeo não encontrado: {video_file}")

if not model_dir.exists():
    raise FileNotFoundError(f"Modelo não encontrado: {model_dir}")

output_txt = video_file.with_suffix(".txt")
output_srt = video_file.with_suffix(".srt")

# =========================================
# EXECUÇÃO
# =========================================
inicio = time.time()

print("Carregando modelo local...")
model, device_used, compute_type_used = load_model_with_fallback(str(model_dir))

print("Iniciando transcrição...")
segments, info = model.transcribe(
    str(video_file),
    language=LANGUAGE,
    beam_size=BEAM_SIZE,
    vad_filter=USE_VAD
)

segments = list(segments)

print("Gerando TXT...")
with open(output_txt, "w", encoding="utf-8") as f:
    for seg in segments:
        texto = seg.text.strip()
        if texto:
            f.write(texto + "\n")

print("Gerando SRT...")
with open(output_srt, "w", encoding="utf-8") as f:
    for i, seg in enumerate(segments, start=1):
        texto = seg.text.strip()
        if texto:
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n")
            f.write(f"{texto}\n\n")

fim = time.time()

print("\nConcluído com sucesso.")
print("Dispositivo usado:", device_used)
print("Compute type usado:", compute_type_used)
print("Idioma detectado:", info.language)
print("Probabilidade:", info.language_probability)
print("TXT:", output_txt)
print("SRT:", output_srt)
print(f"Tempo total: {(fim - inicio)/60:.2f} minutos")