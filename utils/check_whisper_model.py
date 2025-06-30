import platform
import shutil
import psutil
import torch
import GPUtil

def get_gpu_info():
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            return {
                "name": gpu.name,
                "memory_total": gpu.memoryTotal,
                "cuda_available": True
            }
    return {
        "name": "None",
        "memory_total": 0,
        "cuda_available": False
    }

def recommend_whisper_model(cpu_cores, ram_gb, gpu_memory, cuda_available):
    if cuda_available:
        if gpu_memory >= 12:
            return "large"
        elif gpu_memory >= 8:
            return "medium"
        elif gpu_memory >= 4:
            return "small"
        else:
            return "base"
    else:
        if ram_gb >= 16 and cpu_cores >= 8:
            return "medium"
        elif ram_gb >= 8:
            return "base"
        else:
            return "tiny"

def main():
    print("🔍 Checking system specifications...\n")

    cpu_cores = psutil.cpu_count(logical=True)
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    disk_gb = shutil.disk_usage("/")[2] / (1024 ** 3)
    os_info = platform.platform()
    gpu_info = get_gpu_info()

    print(f"🖥️ OS: {os_info}")
    print(f"💾 RAM: {ram_gb:.2f} GB")
    print(f"🧠 CPU Cores: {cpu_cores}")
    print(f"📀 Free Disk Space: {disk_gb:.2f} GB")
    print(f"🎮 GPU: {gpu_info['name']}")
    print(f"🚀 GPU Memory: {gpu_info['memory_total']} MB")
    print(f"⚙️ CUDA Available: {gpu_info['cuda_available']}\n")

    model = recommend_whisper_model(cpu_cores, ram_gb, gpu_info['memory_total'], gpu_info['cuda_available'])

    print(f"✅ Recommended Whisper Model: **{model}**")

if __name__ == "__main__":
    main()
