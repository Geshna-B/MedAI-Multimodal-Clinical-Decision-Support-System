"""
Multimodal Medical Assistant - HPPCS04 Capstone Project
Main Entry Point for CLI and UI Modes
"""

import os
import sys
import json
from datetime import datetime

def main():
    """
    Main function that can run in two modes:
    1. CLI mode: Process all 5 cases automatically
    2. UI mode: Launch the Streamlit dashboard
    """
    if len(sys.argv) == 3:
        # CLI Mode - Process all cases
        run_cli_mode(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        # UI Mode - Launch dashboard
        run_ui_mode()
    else:
        print("Usage:")
        print("  CLI Mode: python main.py <groq_api_key> <gemini_api_key>")
        print("  UI Mode:  python main.py")
        sys.exit(1)

def run_cli_mode(groq_key, gemini_key):
    """Run automated processing of all 5 cases"""
    print("🚀 Starting Multimodal Medical Assistant - CLI Mode")
    print("Processing 5 medical cases...")
    
    try:
        # Import and use your existing MedicalAssistant class
        from medical_assistant import MedicalAssistant
        
        assistant = MedicalAssistant(groq_key, gemini_key)
        successful_cases = 0
        
        for case_num in range(1, 6):
            try:
                print(f"\n📋 Processing Case {case_num}...")
                
                # Use files from current directory (no subdirectories)
                image_path = f"case{case_num}_image.png"
                prescription_path = f"case{case_num}_prescription.txt"
                
                if not os.path.exists(image_path):
                    print(f"   ❌ Image not found: {image_path}")
                    continue
                if not os.path.exists(prescription_path):
                    print(f"   ❌ Prescription not found: {prescription_path}")
                    continue
                
                # Read prescription
                with open(prescription_path, 'r', encoding='utf-8') as f:
                    prescription_text = f.read().strip()
                
                print(f"   🔍 Processing: {image_path}")
                print(f"   📄 Processing: {prescription_path}")
                
                # Process the case using your existing logic
                result = assistant.process_medical_case(image_path, prescription_text)
                
                # Save JSON output in current directory
                output_file = f"case_{case_num}_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                processing_time = result.get('processing_time_seconds', 0)
                print(f"   ✅ Case {case_num} completed in {processing_time}s")
                print(f"   💾 Output saved: {output_file}")
                
                successful_cases += 1
                
            except Exception as e:
                print(f"   ❌ Case {case_num} failed: {str(e)}")
        
        # Final summary
        print("\n" + "=" * 70)
        print(f"🎯 MULTIMODAL PROCESSING COMPLETE: {successful_cases}/5 cases")
        print("📁 Outputs: case_*_result.json")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")

def run_ui_mode():
    """Launch the Streamlit dashboard"""
    try:
        import subprocess
        import webbrowser
        import time
        
        print("🚀 Launching Medical Assistant Dashboard...")
        print("📊 Starting Streamlit server...")
        
        # Check if API keys were provided to main.py
        groq_key = input("Enter Groq API Key (or press Enter to skip): ").strip()
        gemini_key = input("Enter Gemini API Key (or press Enter to skip): ").strip()
        
        if groq_key and gemini_key:
            # Start Streamlit with API keys as arguments
            process = subprocess.Popen([
                'streamlit', 'run', 'app_enhanced.py', 
                groq_key, gemini_key,
                '--server.port', '8501',
                '--server.headless', 'true'
            ])
        else:
            # Start without API keys (user will enter in UI)
            process = subprocess.Popen([
                'streamlit', 'run', 'app_enhanced.py',
                '--server.port', '8501', 
                '--server.headless', 'true'
            ])
        
        # Wait for server to start
        time.sleep(3)
        
        # Open browser automatically
        webbrowser.open('http://localhost:8501')
        
        print("✅ Dashboard launched! Opening browser...")
        print("💡 API keys can be entered in the sidebar if not provided")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Keep running
        process.wait()
        
    except Exception as e:
        print(f"❌ Failed to launch UI: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")


if __name__ == "__main__":
    main()
