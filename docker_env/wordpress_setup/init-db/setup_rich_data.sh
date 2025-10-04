#!/bin/bash

echo "🚀 插入WordPress丰富示例数据..."

# Wait for MySQL to be ready
echo "⏳ 等待MySQL准备就绪..."
sleep 5

# Execute the rich data SQL script
echo "📝 执行WordPress丰富数据脚本..."
docker exec -i wordpress-db mysql -u root -ppassword wordpress < wordpress_setup/init-db/rich_demo_data.sql

if [ $? -eq 0 ]; then
    echo "✅ WordPress丰富数据插入成功！"
    echo "🌐 您现在可以访问: http://localhost:8081"
    echo "📊 插入的丰富数据包括:"
    echo "   - 10篇详细文章 (技术、科学、商业、健康、生活等)"
    echo "   - 4个页面 (关于我们、联系我们、隐私政策、服务条款)"
    echo "   - 10个分类 (技术、科学、商业、健康、旅行、美食、运动、娱乐、教育、生活)"
    echo "   - 15个标签 (编程、网页开发、数据科学等)"
    echo "   - 12条评论"
    echo "   - 导航菜单"
else
    echo "❌ WordPress丰富数据插入失败"
    exit 1
fi


