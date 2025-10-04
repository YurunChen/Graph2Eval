#!/usr/bin/env python3
"""
自动完成WordPress初始设置的脚本
"""

import requests
import time
import re
from urllib.parse import urljoin

def setup_wordpress():
    """自动完成WordPress初始设置"""
    
    base_url = "http://localhost:8081"
    session = requests.Session()
    
    print("🚀 开始自动设置WordPress...")
    
    try:
        # 步骤1：访问安装页面
        print("📋 步骤1：访问安装页面...")
        response = session.get(f"{base_url}/wp-admin/install.php")
        
        if response.status_code != 200:
            print(f"❌ 无法访问安装页面: {response.status_code}")
            return False
        
        # 步骤2：获取nonce值
        print("🔐 步骤2：获取安全令牌...")
        nonce_match = re.search(r'name="_wpnonce" value="([^"]+)"', response.text)
        if not nonce_match:
            print("❌ 无法获取安全令牌")
            return False
        
        nonce = nonce_match.group(1)
        
        # 步骤3：提交安装表单
        print("📝 步骤3：提交安装信息...")
        install_data = {
            'weblog_title': 'My Test Site',
            'user_name': 'admin',
            'admin_password': 'admin123',
            'admin_password2': 'admin123',
            'admin_email': 'admin@example.com',
            'Submit': 'Install WordPress',
            '_wpnonce': nonce,
            '_wp_http_referer': '/wp-admin/install.php?step=2'
        }
        
        response = session.post(f"{base_url}/wp-admin/install.php?step=2", data=install_data)
        
        if response.status_code == 200 and "WordPress has been installed" in response.text:
            print("✅ WordPress安装成功！")
            print(f"🌐 网站地址: {base_url}")
            print(f"🔧 管理后台: {base_url}/wp-admin/")
            print(f"👤 用户名: admin")
            print(f"🔑 密码: admin123")
            return True
        else:
            print("❌ WordPress安装失败")
            return False
            
    except Exception as e:
        print(f"❌ 设置过程中出错: {e}")
        return False

if __name__ == "__main__":
    print("⏳ 等待WordPress启动...")
    time.sleep(5)
    
    if setup_wordpress():
        print("\n🎉 WordPress设置完成！现在可以插入示例数据了。")
    else:
        print("\n❌ WordPress设置失败，请手动完成设置。")


