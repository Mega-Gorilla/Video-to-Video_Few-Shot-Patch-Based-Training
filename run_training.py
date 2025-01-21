# run_training.py
import subprocess
import webbrowser
import time
import sys
from pathlib import Path
import atexit
import netifaces

def get_lan_ip() -> str:
    """Get the LAN IP address of this machine"""
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                ip = addr['addr']
                # Skip localhost and docker interfaces
                if not ip.startswith(('127.', '172.')):
                    return ip
    return 'localhost'

def start_tensorboard(logdir: str, port: int = 6006):
    """Start TensorBoard process"""
    # Ensure log directory exists
    Path(logdir).mkdir(parents=True, exist_ok=True)

    lan_ip = get_lan_ip()
    print(f"Starting TensorBoard...")
    
    tensorboard_process = subprocess.Popen(
        [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", logdir,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--reload_interval", "5"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Register cleanup function
    atexit.register(lambda: tensorboard_process.terminate() if tensorboard_process.poll() is None else None)
    
    time.sleep(3)  # Wait for TensorBoard to start
    return tensorboard_process, lan_ip

def main():
    log_dir = "outputs/logs"
    port = 6006
    
    # Start TensorBoard
    tensorboard_process, lan_ip = start_tensorboard(log_dir, port)
    
    # Open browser
    url = f"http://{lan_ip}:{port}"
    print(f"\nTensorBoard is available at: {url}")
    print(f"Or use: http://localhost:{port}\n")
    webbrowser.open(url)
    
    # Start training
    print("Starting training...")
    subprocess.run([sys.executable, "train.py"])

if __name__ == "__main__":
    main()