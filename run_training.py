import subprocess
import webbrowser
import time
import os
from pathlib import Path
import signal
import atexit
import socket

def get_ip_address():
    """Get local IP address"""
    try:
        # Create a socket object to connect to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # We don't actually connect, but this helps us get the local IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

def start_tensorboard(logdir: str, port: int = 6006) -> subprocess.Popen:
    """
    Start TensorBoard process with LAN access enabled
    
    Args:
        logdir: Directory containing TensorBoard logs
        port: Port number for TensorBoard server
    
    Returns:
        TensorBoard process
    """
    tensorboard_process = subprocess.Popen(
        [
            "tensorboard",
            "--logdir", logdir,
            "--port", str(port),
            "--host", "0.0.0.0",  # Allow connections from any IP
            "--bind_all"  # Bind to all network interfaces
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Register cleanup function
    def cleanup():
        if tensorboard_process.poll() is None:
            tensorboard_process.terminate()
            tensorboard_process.wait()
    
    atexit.register(cleanup)
    
    # Wait for TensorBoard to start
    time.sleep(3)
    
    return tensorboard_process

def open_browser(port: int = 6006, local_only: bool = True):
    """Open web browser to TensorBoard page"""
    if local_only:
        url = f"http://localhost:{port}"
    else:
        ip = get_ip_address()
        url = f"http://{ip}:{port}"
    webbrowser.open(url)

def main():
    PORT = 6006
    # Define log directory
    log_dir = Path("outputs/logs")
    
    # Get local IP address
    ip_address = get_ip_address()
    
    # Start TensorBoard
    print("Starting TensorBoard...")
    tensorboard_process = start_tensorboard(str(log_dir), PORT)
    
    # Print access information
    print(f"\nTensorBoard is now accessible at:")
    print(f"Local: http://localhost:{PORT}")
    print(f"LAN: http://{ip_address}:{PORT}")
    print("\nShare the LAN address with others on the same network to allow them to view TensorBoard.")
    
    # Open browser locally
    print("\nOpening TensorBoard in local browser...")
    open_browser(PORT, local_only=True)
    
    try:
        # Start training
        print("\nStarting training...")
        training_process = subprocess.Popen(
            ["python", "train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Stream output from training process
        while True:
            output = training_process.stdout.readline()
            if output == b'' and training_process.poll() is not None:
                break
            if output:
                print(output.decode().strip())
                
        # Check for errors
        if training_process.returncode != 0:
            print("Training failed!")
            for line in training_process.stderr.readlines():
                print(line.decode().strip())
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
        
    finally:
        # Cleanup processes
        if tensorboard_process.poll() is None:
            tensorboard_process.terminate()
            tensorboard_process.wait()

if __name__ == "__main__":
    main()