import os

def running_in_docker():
    # Method 1: buscar archivo /.dockerenv
    if os.path.exists('/.dockerenv'):
        return True
    # Method 2: revisar contenido de cgroup
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            return any('docker' in line or 'containerd' in line for line in f)
    except FileNotFoundError:
        return False