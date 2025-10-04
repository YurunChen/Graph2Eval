#!/usr/bin/env python3
"""
Remove Malicious Elements from SuiteCRM
Removes malicious elements from SuiteCRM PHP files
"""

import subprocess
from loguru import logger


def remove_malicious_elements():
    """Remove all malicious elements from SuiteCRM files"""
    container_name = "suitecrm_setup-suitecrm-1"
    
    print("🧹 Removing malicious elements from SuiteCRM...")
    
    # Find all files containing malicious elements
    find_command = f"docker exec {container_name} find /opt/bitnami/suitecrm -name '*.php' -exec grep -l 'malicious_' {{}} \\;"
    
    try:
        result = subprocess.run(find_command, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            files_with_malicious = result.stdout.strip().split('\n')
            print(f"📁 Found {len(files_with_malicious)} files with malicious elements")
            
            for file_path in files_with_malicious:
                if file_path:
                    print(f"🧹 Cleaning {file_path}...")
                    
                    # Remove lines containing malicious elements
                    sed_command = f"docker exec {container_name} sed -i '/malicious_/d' {file_path}"
                    subprocess.run(sed_command, shell=True, check=True)
                    
                    print(f"✅ Cleaned {file_path}")
        else:
            print("✅ No malicious elements found")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error removing malicious elements: {e}")
        return False
    
    # Verify removal
    print("\n🔍 Verifying removal...")
    verify_command = f"docker exec {container_name} find /opt/bitnami/suitecrm -name '*.php' -exec grep -l 'malicious_' {{}} \\;"
    
    try:
        result = subprocess.run(verify_command, shell=True, capture_output=True, text=True)
        if result.returncode == 1 or not result.stdout.strip():
            print("✅ All malicious elements have been successfully removed!")
            return True
        else:
            print(f"⚠️ Some malicious elements may still exist: {result.stdout}")
            return False
    except Exception as e:
        print(f"❌ Error verifying removal: {e}")
        return False


def main():
    """Main function"""
    print("🔒 SuiteCRM Malicious Element Remover")
    print("=" * 50)
    
    success = remove_malicious_elements()
    
    if success:
        print("\n🎉 Successfully removed all malicious elements!")
        print("🌐 SuiteCRM is now clean and safe to use")
    else:
        print("\n❌ Failed to remove all malicious elements")
        print("Please check manually or restart the container")


if __name__ == "__main__":
    main()
