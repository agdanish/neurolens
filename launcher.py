import os
import sys
import subprocess

def run_diagnostic_system():
    """
    Robust Launcher for NeuroFundus System
    Sets up the 'Legacy Keras' environment sandbox before starting the UI.
    """
    # 1. FORCE the Environment Variables
    # This acts as the "Nuclear Fix" but does it cleanly from Python
    env_copy = os.environ.copy()
    env_copy["TF_USE_LEGACY_KERAS"] = "1"
    env_copy["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    # 2. Define the Dashboard File
    # We will name the main app file 'neuro_dashboard.py'
    dashboard_file = "neuro_dashboard.py"
    
    # Check if dashboard exists
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: Could not find {dashboard_file} in the current folder.")
        return

    # 3. Construct the Launch Command
    # This runs: "python -m streamlit run neuro_dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_file]
    
    print("="*60)
    print("🧠 NeuroFundus AI Launcher")
    print("="*60)
    print("🔧 Configuring Deep Learning Environment...")
    print("   [✓] Legacy Keras Mode: ACTIVATED")
    print("   [✓] OneDNN Optimizations: DISABLED")
    print("🚀 Launching Dashboard Interface...")
    print("="*60)
    
    # 4. Execute the Dashboard in the Sandbox
    try:
        subprocess.check_call(cmd, env=env_copy)
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested.")
    except Exception as e:
        print(f"\n❌ Critical Launch Error: {e}")

if __name__ == "__main__":
    run_diagnostic_system()