#!/usr/bin/env python3
"""
Upload Brello EI 0 to Hugging Face
Created by Epic Systems | Engineered by Rehan Temkar
"""

import subprocess
import sys
import os

def main():
    """Upload Brello EI 0 to Hugging Face"""
    
    print("🤖 Uploading Brello EI 0 to Hugging Face")
    print("Created by Epic Systems | Engineered by Rehan Temkar")
    print("=" * 50)
    
    # Check if huggingface-cli is installed
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
        print("✅ huggingface-cli found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ huggingface-cli not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    
    # Get username
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ Username is required.")
        return
    
    # Create repository
    repo_name = "brello-ei-0"
    full_repo_name = f"{username}/{repo_name}"
    
    print(f"📁 Creating repository: {full_repo_name}")
    try:
        subprocess.run([
            "huggingface-cli", "repo", "create", 
            repo_name, "--type", "model"
        ], check=True)
        print("✅ Repository created successfully!")
    except subprocess.CalledProcessError:
        print(f"⚠️  Repository {repo_name} might already exist.")
    
    # Upload model files
    print("📤 Uploading model files...")
    try:
        subprocess.run([
            "huggingface-cli", "upload", full_repo_name,
            ".", "--include", "*.py,*.md,*.txt"
        ], check=True)
        print("✅ Model files uploaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        return
    
    # Ask about logo
    logo_choice = input("\n🎨 Do you have a logo.png file to upload? (y/n): ").strip().lower()
    if logo_choice == 'y':
        logo_path = input("Enter path to your logo.png file: ").strip()
        if os.path.exists(logo_path):
            try:
                subprocess.run([
                    "huggingface-cli", "upload", full_repo_name, logo_path
                ], check=True)
                print("✅ Logo uploaded successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Logo upload failed: {e}")
        else:
            print("❌ Logo file not found.")
    
    print(f"\n🎉 Your model is now available at: https://huggingface.co/{full_repo_name}")
    print("\n📋 Next steps:")
    print("1. Visit your model page on Hugging Face")
    print("2. Add a logo if you haven't already")
    print("3. Update the model description")
    print("4. Share your model with the community!")
    print("\n💡 For logo creation help, see: logo_creation_guide.md")

if __name__ == "__main__":
    main()
