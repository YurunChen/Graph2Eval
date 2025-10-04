#!/bin/bash

echo "🧹 清理WordPress重复数据..."

# Execute the cleanup SQL script
echo "📝 执行重复数据清理脚本..."
docker exec -i wordpress-db mysql -u root -ppassword wordpress < wordpress_setup/init-db/cleanup_duplicates.sql

if [ $? -eq 0 ]; then
    echo "✅ 重复数据清理完成！"
    echo "📊 清理后的数据统计:"
    echo "   - 文章数量已优化"
    echo "   - 页面数量已优化"
    echo "   - 评论数量已优化"
    echo "   - 分类和标签已去重"
    echo "🌐 您现在可以访问: http://localhost:8081"
else
    echo "❌ 重复数据清理失败"
    exit 1
fi


