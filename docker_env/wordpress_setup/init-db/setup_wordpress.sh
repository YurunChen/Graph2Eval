#!/bin/bash

echo "🚀 Setting up WordPress database with demo data..."

# Wait for MySQL to be ready
echo "⏳ Waiting for MySQL to be ready..."
sleep 10

# Execute the SQL script
echo "📝 Executing WordPress demo data script..."
docker exec -i wordpress-db mysql -u root -ppassword wordpress < wordpress_setup/init-db/simple_demo_data.sql

if [ $? -eq 0 ]; then
    echo "✅ WordPress database setup completed successfully!"
    echo "🌐 You can now access WordPress at: http://localhost:8081"
    echo "📊 Demo data includes:"
    echo "   - 3 sample posts (Python, AI, Health)"
    echo "   - 2 sample pages (About Us, Contact Us)"
    echo "   - 3 sample comments"
else
    echo "❌ Failed to setup WordPress database"
    exit 1
fi
