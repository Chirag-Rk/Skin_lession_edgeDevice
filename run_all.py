import sys
import subprocess
import time

print("="*60)
print("  FEDERATED LEARNING - MULTI-PROCESS LAUNCHER")
print("="*60)

# Start Server
print("[Launcher] Starting FL Server...")
server = subprocess.Popen([sys.executable, "fl/server.py"])
time.sleep(3) # Give server time to boot

clients = []
for i in range(3):
    print(f"[Launcher] Starting FL Client {i}...")
    proc = subprocess.Popen([sys.executable, "fl/client_app.py", str(i)])
    clients.append(proc)

print("[Launcher] All processes launched successfully!")
print("[Launcher] Waiting for Federated Learning rounds to complete...\n")

# Wait for server and clients
for proc in clients:
    proc.wait()
server.wait()

print("\n" + "="*60)
print("[Launcher] FEDERATED LEARNING COMPLETE.")
print("="*60)
