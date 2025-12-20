"""
GPU Monitoring and Optimization Script
Real-time GPU monitoring, memory optimization, and performance tuning
"""

import torch
import time
import psutil
import subprocess
from typing import Dict, List, Tuple
from datetime import datetime
import json
from pathlib import Path

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("pynvml not available. Install with: pip install pynvml")


class GPUMonitor:
    """Real-time GPU monitoring and optimization"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history = []
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except:
                self.nvml_initialized = False
                print("Could not initialize NVML")
        else:
            self.nvml_initialized = False
    
    def get_gpu_info(self, device_id: int = 0) -> Dict:
        """Get comprehensive GPU information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        info = {
            "available": True,
            "device_id": device_id,
            "device_name": torch.cuda.get_device_name(device_id),
            "compute_capability": torch.cuda.get_device_capability(device_id),
        }
        
        # PyTorch memory info
        info["pytorch"] = {
            "allocated_gb": torch.cuda.memory_allocated(device_id) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(device_id) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(device_id) / 1e9,
        }
        
        # NVML detailed info
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info["nvml_memory"] = {
                    "total_gb": mem_info.total / 1e9,
                    "used_gb": mem_info.used / 1e9,
                    "free_gb": mem_info.free / 1e9,
                    "utilization_pct": (mem_info.used / mem_info.total) * 100
                }
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info["utilization"] = {
                    "gpu_pct": util.gpu,
                    "memory_pct": util.memory
                }
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                info["temperature_c"] = temp
                
                # Power
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                info["power"] = {
                    "usage_w": power_usage,
                    "limit_w": power_limit,
                    "utilization_pct": (power_usage / power_limit) * 100
                }
                
                # Clock speeds
                clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                info["clocks_mhz"] = {
                    "sm": clock_sm,
                    "memory": clock_mem
                }
                
            except Exception as e:
                info["nvml_error"] = str(e)
        
        return info
    
    def print_gpu_status(self, device_id: int = 0):
        """Print formatted GPU status"""
        info = self.get_gpu_info(device_id)
        
        if not info["available"]:
            print("No GPU available")
            return
        
        print("\n" + "="*70)
        print(f"GPU STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        print(f"\nDevice: {info['device_name']}")
        print(f"   Compute Capability: {info['compute_capability']}")
        
        # PyTorch Memory
        print(f"\nPyTorch Memory:")
        print(f"   Allocated: {info['pytorch']['allocated_gb']:.2f} GB")
        print(f"   Reserved:  {info['pytorch']['reserved_gb']:.2f} GB")
        print(f"   Peak:      {info['pytorch']['max_allocated_gb']:.2f} GB")
        
        # NVML Info
        if "nvml_memory" in info:
            mem = info["nvml_memory"]
            print(f"\nGPU Memory:")
            print(f"   Total: {mem['total_gb']:.2f} GB")
            print(f"   Used:  {mem['used_gb']:.2f} GB ({mem['utilization_pct']:.1f}%)")
            print(f"   Free:  {mem['free_gb']:.2f} GB")
            
            # Memory bar
            bar_width = 40
            used_bars = int((mem['utilization_pct'] / 100) * bar_width)
            bar = "█" * used_bars + "░" * (bar_width - used_bars)
            print(f"   [{bar}]")
        
        if "utilization" in info:
            util = info["utilization"]
            print(f"\nUtilization:")
            print(f"   GPU:    {util['gpu_pct']}%")
            print(f"   Memory: {util['memory_pct']}%")
        
        if "temperature_c" in info:
            temp = info["temperature_c"]
            temp_status = "HIGH" if temp > 80 else "NORMAL" if temp > 60 else "LOW"
            print(f"\nTemperature: {temp}°C ({temp_status})")
        
        if "power" in info:
            power = info["power"]
            print(f"\nPower:")
            print(f"   Usage: {power['usage_w']:.1f}W / {power['limit_w']:.1f}W")
            print(f"   Utilization: {power['utilization_pct']:.1f}%")
        
        if "clocks_mhz" in info:
            clocks = info["clocks_mhz"]
            print(f"\nClock Speeds:")
            print(f"   SM:     {clocks['sm']} MHz")
            print(f"   Memory: {clocks['memory']} MHz")
        
        print("="*70 + "\n")
    
    def optimize_memory(self, device_id: int = 0):
        """Optimize GPU memory usage"""
        print("\nOptimizing GPU memory...")
        
        # Get initial state
        initial_allocated = torch.cuda.memory_allocated(device_id) / 1e9
        initial_reserved = torch.cuda.memory_reserved(device_id) / 1e9
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get final state
        final_allocated = torch.cuda.memory_allocated(device_id) / 1e9
        final_reserved = torch.cuda.memory_reserved(device_id) / 1e9
        
        # Calculate savings
        freed_allocated = initial_allocated - final_allocated
        freed_reserved = initial_reserved - final_reserved
        
        print(f"Memory optimization complete:")
        print(f"   Freed allocated memory: {freed_allocated:.2f} GB")
        print(f"   Freed reserved memory:  {freed_reserved:.2f} GB")
        print(f"   Current allocated: {final_allocated:.2f} GB")
        print(f"   Current reserved:  {final_reserved:.2f} GB")
    
    def monitor_continuously(self, interval: int = 1, duration: int = 60):
        """Monitor GPU continuously"""
        print(f"\nMonitoring GPU for {duration} seconds (interval: {interval}s)")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        self.monitoring = True
        
        try:
            while self.monitoring and (time.time() - start_time) < duration:
                info = self.get_gpu_info()
                
                if info["available"]:
                    # Store metrics
                    metrics = {
                        "timestamp": datetime.now().isoformat(),
                        "allocated_gb": info["pytorch"]["allocated_gb"],
                        "reserved_gb": info["pytorch"]["reserved_gb"],
                    }
                    
                    if "utilization" in info:
                        metrics["gpu_util_pct"] = info["utilization"]["gpu_pct"]
                        metrics["mem_util_pct"] = info["utilization"]["memory_pct"]
                    
                    if "temperature_c" in info:
                        metrics["temperature_c"] = info["temperature_c"]
                    
                    self.metrics_history.append(metrics)
                    
                    # Print compact status
                    mem_alloc = info["pytorch"]["allocated_gb"]
                    gpu_util = info.get("utilization", {}).get("gpu_pct", 0)
                    temp = info.get("temperature_c", 0)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Memory: {mem_alloc:.2f}GB | "
                          f"GPU: {gpu_util}% | "
                          f"Temp: {temp}°C")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        self.monitoring = False
        print(f"\nMonitoring complete. Collected {len(self.metrics_history)} data points")
    
    def save_metrics(self, filepath: str = "gpu_metrics.json"):
        """Save metrics history to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")
    
    def get_optimization_recommendations(self, device_id: int = 0) -> List[str]:
        """Get GPU optimization recommendations"""
        info = self.get_gpu_info(device_id)
        recommendations = []
        
        if not info["available"]:
            return ["No GPU available"]
        
        # Memory recommendations
        if "nvml_memory" in info:
            mem_util = info["nvml_memory"]["utilization_pct"]
            
            if mem_util > 95:
                recommendations.append(f"CRITICAL: GPU memory at {mem_util:.1f}%. Reduce batch size or model size!")
            elif mem_util > 85:
                recommendations.append(f"WARNING: GPU memory at {mem_util:.1f}%. Consider reducing context length or batch size.")
            elif mem_util < 50:
                recommendations.append(f"GOOD: GPU memory at {mem_util:.1f}%. You can increase batch size for better performance.")
        
        # Utilization recommendations
        if "utilization" in info:
            gpu_util = info["utilization"]["gpu_pct"]
            
            if gpu_util < 30:
                recommendations.append(f"Low GPU utilization ({gpu_util:.1f}%). Increase batch size or check for CPU bottlenecks.")
            elif gpu_util > 95:
                recommendations.append(f"Excellent GPU utilization ({gpu_util:.1f}%)!")
        
        # Temperature recommendations
        if "temperature_c" in info:
            temp = info["temperature_c"]
            
            if temp > 85:
                recommendations.append(f"HIGH TEMPERATURE: {temp}°C. Check cooling and reduce workload.")
            elif temp > 75:
                recommendations.append(f"Temperature at {temp}°C. Monitor cooling.")
            elif temp < 60:
                recommendations.append(f"Temperature good at {temp}°C.")
        
        # Power recommendations
        if "power" in info:
            power_util = info["power"]["utilization_pct"]
            
            if power_util > 95:
                recommendations.append(f"Power limit reached ({power_util:.1f}%). Consider increasing power limit.")
        
        return recommendations if recommendations else ["No optimization needed. GPU running optimally!"]
    
    def benchmark_memory(self, sizes: List[int] = None) -> Dict:
        """Benchmark memory usage with different tensor sizes"""
        if sizes is None:
            sizes = [1024, 2048, 4096, 8192, 16384]
        
        print("\nRunning memory benchmark...")
        results = {}
        
        for size in sizes:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1e9
            
            try:
                # Create tensor
                tensor = torch.randn(size, size).cuda()
                
                # Measure
                final_memory = torch.cuda.memory_allocated() / 1e9
                memory_used = final_memory - initial_memory
                
                results[size] = {
                    "memory_gb": memory_used,
                    "success": True
                }
                
                print(f"  Size {size:5d}x{size:5d}: {memory_used:.3f} GB")
                
                # Clean up
                del tensor
                
            except RuntimeError as e:
                results[size] = {
                    "memory_gb": 0,
                    "success": False,
                    "error": str(e)
                }
                print(f"  Size {size:5d}x{size:5d}: FAILED (OOM)")
        
        torch.cuda.empty_cache()
        return results
    
    def __del__(self):
        """Cleanup"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def interactive_menu():
    """Interactive GPU monitoring menu"""
    monitor = GPUMonitor()
    
    while True:
        print("\n" + "="*70)
        print("GPU MONITORING & OPTIMIZATION MENU")
        print("="*70)
        print("\n1. Show GPU Status")
        print("2. Optimize Memory")
        print("3. Monitor Continuously")
        print("4. Benchmark Memory")
        print("5. Get Optimization Recommendations")
        print("6. Save Metrics")
        print("7. Clear GPU Cache")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            monitor.print_gpu_status()
        
        elif choice == "2":
            monitor.optimize_memory()
        
        elif choice == "3":
            duration = input("Duration in seconds (default: 60): ").strip()
            duration = int(duration) if duration else 60
            monitor.monitor_continuously(interval=1, duration=duration)
        
        elif choice == "4":
            results = monitor.benchmark_memory()
        
        elif choice == "5":
            recommendations = monitor.get_optimization_recommendations()
            print("\nOptimization Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
        
        elif choice == "6":
            filepath = input("Filepath (default: gpu_metrics.json): ").strip()
            filepath = filepath if filepath else "gpu_metrics.json"
            monitor.save_metrics(filepath)
        
        elif choice == "7":
            torch.cuda.empty_cache()
            print("GPU cache cleared")
        
        elif choice == "8":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-8.")


def main():
    """Main function"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   GPU Monitoring & Optimization Tool                   ║
    ║   Real-time GPU monitoring and memory optimization     ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("No GPU detected!")
        print("\nTroubleshooting:")
        print("  1. Check NVIDIA drivers: nvidia-smi")
        print("  2. Check CUDA installation: nvcc --version")
        print("  3. Reinstall PyTorch with CUDA support")
        return
    
    # Show quick status
    monitor = GPUMonitor()
    monitor.print_gpu_status()
    
    # Show recommendations
    recommendations = monitor.get_optimization_recommendations()
    print("Quick Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")
    
    # Start interactive menu
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()

