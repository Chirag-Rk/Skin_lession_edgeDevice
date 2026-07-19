import subprocess
import time
import os
import sys

# =============================================================================
# CONFIGURATION (FAST MODE)
# =============================================================================
ROUNDS = "1"
CLIENTS = "2"
SPLIT_MODE = "iid"  # Use IID for fast verification
PYTHON_EXE = sys.executable

print("\n" + "="*70)
print("  AUTOMATED SKIN CANCER FEDERATED LEARNING PIPELINE")
print("="*70)

# =============================================================================
# STEP 1: START SERVER
# =============================================================================
print(f"\n[Step 1] Launching FL Server (localhost:8080)...")
env = os.environ.copy()
env["PYTHONPATH"] = "."
env["ROUNDS"] = ROUNDS
env["CLIENTS"] = CLIENTS
env["SPLIT_MODE"] = SPLIT_MODE

server_proc = subprocess.Popen(
    [PYTHON_EXE, "fl/server.py"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Wait for server to start
time.sleep(5)
if server_proc.poll() is not None:
    print("[Error] Server failed to start. Output:")
    print(server_proc.stdout.read())
    sys.exit(1)

print("[OK] Server active.")

# =============================================================================
# STEP 2: START CLIENTS
# =============================================================================
client_procs = []
for i in range(int(CLIENTS)):
    print(f"[Step 2] Launching Client {i}...")
    p = subprocess.Popen(
        [PYTHON_EXE, "fl/client_app.py", str(i)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    client_procs.append(p)
    time.sleep(2)

# =============================================================================
# STEP 3: MONITOR & WAIT
# =============================================================================
print("\n[Step 3] Training in progress. Please wait...")
print("  (FL Server will close automatically after 1 round)")

# Wait for server to finish
server_out, _ = server_proc.communicate()
print("\n--- SERVER LOG SUMMARY ---")
print(server_out[-1000:] if len(server_out) > 1000 else server_out)
print("--------------------------")

# Cleanup clients
for p in client_procs:
    p.terminate()

print("\n✅ Federated Learning phase complete.")

# =============================================================================
# STEP 4: RUN DOWNSTREAM PIPELINE
# =============================================================================
print("\n[Step 4] Running Downstream Analysis...")

def run_script(name, cmd_list):
    print(f"  → Running {name}...")
    try:
        subprocess.run(cmd_list, env=env, check=True, capture_output=True, text=True)
        print(f"    [OK] {name} Success.")
    except subprocess.CalledProcessError as e:
        print(f"    [Warning] {name} Failed: {e.stderr}")

run_script("Centralized Training", [PYTHON_EXE, "centralized_train.py"])
run_script("Metric Visualization", [PYTHON_EXE, "plot_fl.py"])
run_script("Comparison Plotting",  [PYTHON_EXE, "compare.py"])
run_script("Grad-CAM Generation",  [PYTHON_EXE, "gradcam.py", "--n", "2"])

print("\n" + "="*70)
print("  ALL STEPS COMPLETED!")
print("  Artifacts saved to plots/ and logs/")
print("="*70)
