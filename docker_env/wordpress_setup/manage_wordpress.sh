#!/bin/bash

# WordPress Management Script
# Usage: ./manage_wordpress.sh [start|stop|restart|status|logs|clean]

NETWORK_NAME="wordpress-network"
DB_CONTAINER="wordpress-db"
WP_CONTAINER="wordpress"

case "$1" in
    start)
        echo "Starting WordPress environment..."
        
        # Create network if it doesn't exist
        if ! docker network ls | grep -q $NETWORK_NAME; then
            echo "Creating network: $NETWORK_NAME"
            docker network create $NETWORK_NAME
        fi
        
        # Start MySQL database
        if ! docker ps -a | grep -q $DB_CONTAINER; then
            echo "Creating MySQL database container..."
            docker run --name $DB_CONTAINER --network $NETWORK_NAME \
                -e MYSQL_ROOT_PASSWORD=somewordpress \
                -e MYSQL_DATABASE=wordpress \
                -e MYSQL_USER=wordpress \
                -e MYSQL_PASSWORD=wordpress \
                -d mysql:8.0
        else
            echo "Starting MySQL database container..."
            docker start $DB_CONTAINER
        fi
        
        # Wait for MySQL to be ready
        echo "Waiting for MySQL to be ready..."
        sleep 10
        
        # Start WordPress
        if ! docker ps -a | grep -q $WP_CONTAINER; then
            echo "Creating WordPress container..."
            docker run --name $WP_CONTAINER --network $NETWORK_NAME \
                -e WORDPRESS_DB_HOST=$DB_CONTAINER:3306 \
                -e WORDPRESS_DB_USER=wordpress \
                -e WORDPRESS_DB_PASSWORD=wordpress \
                -e WORDPRESS_DB_NAME=wordpress \
                -e WORDPRESS_DEBUG=1 \
                -p 8081:80 \
                -d wordpress:latest
        else
            echo "Starting WordPress container..."
            docker start $WP_CONTAINER
        fi
        
        echo "WordPress is starting up..."
        echo "Access URL: http://localhost:8081"
        ;;
        
    stop)
        echo "Stopping WordPress environment..."
        docker stop $WP_CONTAINER $DB_CONTAINER 2>/dev/null || true
        echo "WordPress environment stopped."
        ;;
        
    restart)
        echo "Restarting WordPress environment..."
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo "WordPress Environment Status:"
        echo "=============================="
        docker ps -a --filter "name=wordpress" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        docker ps -a --filter "name=wordpress-db" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        echo "Network:"
        docker network ls | grep $NETWORK_NAME || echo "Network $NETWORK_NAME not found"
        ;;
        
    logs)
        echo "WordPress logs:"
        docker logs $WP_CONTAINER
        echo ""
        echo "Database logs:"
        docker logs $DB_CONTAINER
        ;;
        
    clean)
        echo "Cleaning up WordPress environment..."
        docker stop $WP_CONTAINER $DB_CONTAINER 2>/dev/null || true
        docker rm $WP_CONTAINER $DB_CONTAINER 2>/dev/null || true
        docker network rm $NETWORK_NAME 2>/dev/null || true
        echo "WordPress environment cleaned up."
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Start WordPress environment"
        echo "  stop    - Stop WordPress environment"
        echo "  restart - Restart WordPress environment"
        echo "  status  - Show status of containers"
        echo "  logs    - Show container logs"
        echo "  clean   - Remove all containers and network"
        exit 1
        ;;
esac
