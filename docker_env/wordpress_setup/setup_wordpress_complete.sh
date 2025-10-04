#!/bin/bash

echo "🚀 WordPress 完整设置脚本开始..."

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 清理现有容器
echo "🧹 清理现有容器..."
docker stop wordpress wordpress-db 2>/dev/null || true
docker rm wordpress wordpress-db 2>/dev/null || true

# 创建MySQL容器
echo "📦 创建MySQL容器..."
docker run -d --name wordpress-db \
    -e MYSQL_ROOT_PASSWORD=password \
    -e MYSQL_DATABASE=wordpress \
    mysql:8.0

echo "⏳ 等待MySQL启动..."
sleep 20

# 创建WordPress容器
echo "📦 创建WordPress容器..."
docker run -d --name wordpress \
    --link wordpress-db:mysql \
    -p 8081:80 \
    -e WORDPRESS_DB_HOST=wordpress-db \
    -e WORDPRESS_DB_USER=root \
    -e WORDPRESS_DB_PASSWORD=password \
    -e WORDPRESS_DB_NAME=wordpress \
    wordpress:latest

echo "⏳ 等待WordPress启动..."
sleep 15

# 自动设置WordPress
echo "🔧 自动设置WordPress..."
if command -v python3 &> /dev/null; then
    python3 wordpress_setup/auto_setup_wordpress.py
else
    echo "⚠️  Python3未安装，请手动完成WordPress设置"
    echo "   访问: http://localhost:8081"
    echo "   用户名: admin, 密码: admin123"
    read -p "按回车键继续..."
fi

# 插入示例数据
echo "📝 插入示例数据..."
if [ -f "wordpress_setup/init-db/rich_demo_data.sql" ]; then
    docker exec -i wordpress-db mysql -u root -ppassword wordpress < wordpress_setup/init-db/rich_demo_data.sql
fi

# 清理重复数据
echo "🧹 清理重复数据..."
if [ -f "wordpress_setup/init-db/cleanup_duplicates.sql" ]; then
    docker exec -i wordpress-db mysql -u root -ppassword wordpress < wordpress_setup/init-db/cleanup_duplicates.sql
fi

echo ""
echo "✅ WordPress设置完成！"
echo "🌐 网站地址: http://localhost:8081"
echo "🔧 管理后台: http://localhost:8081/wp-admin/"
echo "👤 用户名: admin"
echo "🔑 密码: admin123"
