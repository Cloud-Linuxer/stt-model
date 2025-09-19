#!/usr/bin/env python3
"""
Resource Monitoring Script for Faster-Whisper
"""

import os
import time
import psutil
import subprocess
import threading
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'name': stats[0],
                'memory_used': float(stats[1]),
                'memory_total': float(stats[2]),
                'gpu_util': float(stats[3]),
                'temperature': float(stats[4]),
                'power': float(stats[5]) if stats[5] != '[Not Supported]' else 0
            }
    except:
        pass
    return None

def monitor_resources(stop_event):
    """Monitor system resources in background"""
    max_cpu = 0
    max_ram = 0
    max_gpu_mem = 0
    max_gpu_util = 0

    print("\n📊 Resource Monitoring Started...")
    print("-" * 60)

    while not stop_event.is_set():
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_percent = ram.percent

        # GPU
        gpu_stats = get_gpu_stats()

        # Update maximums
        max_cpu = max(max_cpu, cpu_percent)
        max_ram = max(max_ram, ram_used_gb)

        if gpu_stats:
            max_gpu_mem = max(max_gpu_mem, gpu_stats['memory_used'])
            max_gpu_util = max(max_gpu_util, gpu_stats['gpu_util'])

            print(f"\r💻 CPU: {cpu_percent:5.1f}% (max: {max_cpu:.1f}%) | "
                  f"🧠 RAM: {ram_used_gb:.1f}GB/{ram.total/(1024**3):.1f}GB ({ram_percent:.1f}%) | "
                  f"🎮 GPU: {gpu_stats['gpu_util']:.0f}% | "
                  f"💾 VRAM: {gpu_stats['memory_used']:.0f}MB/{gpu_stats['memory_total']:.0f}MB | "
                  f"🌡️ {gpu_stats['temperature']:.0f}°C | "
                  f"⚡ {gpu_stats['power']:.0f}W", end='', flush=True)
        else:
            print(f"\r💻 CPU: {cpu_percent:5.1f}% (max: {max_cpu:.1f}%) | "
                  f"🧠 RAM: {ram_used_gb:.1f}GB/{ram.total/(1024**3):.1f}GB ({ram_percent:.1f}%)",
                  end='', flush=True)

        time.sleep(0.5)

    print("\n" + "-" * 60)
    print("📊 Peak Resource Usage:")
    print(f"  - Max CPU: {max_cpu:.1f}%")
    print(f"  - Max RAM: {max_ram:.1f} GB")
    if max_gpu_mem > 0:
        print(f"  - Max GPU Utilization: {max_gpu_util:.0f}%")
        print(f"  - Max VRAM: {max_gpu_mem:.0f} MB")

def test_with_monitoring():
    """Test Faster-Whisper with resource monitoring"""

    print("=" * 60)
    print("Faster-Whisper Resource Monitoring Test")
    print("=" * 60)

    # Start monitoring in background
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event,))
    monitor_thread.start()

    try:
        # Load model
        print("\n🔄 Loading Faster-Whisper Large-v3 model...")
        device = "cuda"
        compute_type = "float16"

        start_time = time.time()
        model = WhisperModel(
            "large-v3",
            device=device,
            compute_type=compute_type,
            download_root="/app/models",
            device_index=0,
            num_workers=1,
            cpu_threads=4
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds\n")

        # Test with different audio lengths
        print("🎤 Testing with synthetic audio of different lengths...")

        test_durations = [5, 10, 30, 60]  # seconds

        for duration in test_durations:
            print(f"\n📝 Processing {duration}-second audio...")

            # Create synthetic audio
            sample_rate = 16000
            audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.01

            # Save as temporary file
            import soundfile as sf
            temp_path = f"/tmp/test_audio_{duration}s.wav"
            sf.write(temp_path, audio, sample_rate)

            # Transcribe
            start_time = time.time()
            segments, info = model.transcribe(
                temp_path,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=2000,
                    speech_pad_ms=400
                )
            )

            # Process results
            transcription = ""
            for segment in segments:
                transcription += segment.text

            process_time = time.time() - start_time

            print(f"  ⏱️ Processing time: {process_time:.2f} seconds")
            print(f"  ⚡ Speed: {duration/process_time:.2f}x realtime")

            # Clean up
            os.remove(temp_path)

            # Wait a bit between tests
            time.sleep(2)

        # Test with actual audio file if available
        data_dir = Path("/app/data")
        audio_files = list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))

        if audio_files:
            print(f"\n📂 Processing real audio file: {audio_files[0].name}")

            start_time = time.time()
            segments, info = model.transcribe(
                str(audio_files[0]),
                beam_size=5,
                vad_filter=True
            )

            transcription = ""
            for segment in segments:
                transcription += segment.text

            process_time = time.time() - start_time

            print(f"  ⏱️ Processing time: {process_time:.2f} seconds")
            print(f"  🎧 Audio duration: {info.duration:.2f} seconds")
            print(f"  ⚡ Speed: {info.duration/process_time:.2f}x realtime")
            print(f"  📝 Transcription preview: {transcription[:100]}...")

    finally:
        # Stop monitoring
        print("\n\n🛑 Stopping resource monitor...")
        stop_event.set()
        monitor_thread.join()

    print("\n" + "=" * 60)
    print("Resource monitoring test completed!")
    print("=" * 60)

def main():
    """Main function"""
    try:
        # Check if running in Docker
        if os.path.exists('/.dockerenv'):
            print("🐳 Running in Docker container")
        else:
            print("💻 Running on host system")

        # System info
        print(f"\n📊 System Information:")
        print(f"  - CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        print(f"  - Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

        gpu_stats = get_gpu_stats()
        if gpu_stats:
            print(f"  - GPU: {gpu_stats['name']}")
            print(f"  - VRAM: {gpu_stats['memory_total']:.0f} MB")

        test_with_monitoring()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())