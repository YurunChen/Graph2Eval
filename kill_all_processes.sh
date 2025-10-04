#!/bin/bash

# Script to kill all processes related to GraphEval2/Graph2Eval project
# This script will terminate all running processes including browsers and Python processes

echo "🔍 Searching for GraphEval2/Graph2Eval related processes..."

# Function to kill processes by pattern
kill_processes() {
    local pattern="$1"
    local description="$2"
    
    echo "🔍 Looking for $description..."
    
    # Find processes matching the pattern
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        echo "📋 Found $description processes: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        echo "✅ Killed $description processes"
    else
        echo "ℹ️  No $description processes found"
    fi
}

# Function to kill processes by name
kill_by_name() {
    local name="$1"
    local description="$2"
    
    echo "🔍 Looking for $description..."
    
    # Find processes by name
    local pids=$(pgrep "$name" 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        echo "📋 Found $description processes: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        echo "✅ Killed $description processes"
    else
        echo "ℹ️  No $description processes found"
    fi
}

# Kill main Python processes
kill_processes "benchmark_runner.py" "benchmark runner"
kill_processes "GraphEval2" "GraphEval2 project"
kill_processes "Graph2Eval" "Graph2Eval project"

# Kill browser automation processes
kill_processes "playwright" "Playwright browser automation"
kill_by_name "chromium" "Chromium browser"
kill_by_name "chrome" "Chrome browser"
kill_by_name "firefox" "Firefox browser"
kill_by_name "webkit" "WebKit browser"

# Kill conda environment processes
kill_processes "conda run -n graph2eval" "conda graph2eval environment"

# Kill any remaining Python processes in the project directory
echo "🔍 Looking for Python processes in project directory..."
project_pids=$(ps aux | grep -E "python.*GraphEval|python.*Graph2Eval" | grep -v grep | awk '{print $2}' || true)

if [ -n "$project_pids" ]; then
    echo "📋 Found additional Python processes: $project_pids"
    echo "$project_pids" | xargs kill -9 2>/dev/null || true
    echo "✅ Killed additional Python processes"
else
    echo "ℹ️  No additional Python processes found"
fi

# Final verification
echo ""
echo "🔍 Final verification - checking for remaining processes..."

remaining_processes=$(ps aux | grep -E "(benchmark_runner|GraphEval|Graph2Eval|playwright|chromium)" | grep -v grep | grep -v "kill_all_processes" || true)

if [ -n "$remaining_processes" ]; then
    echo "⚠️  Some processes may still be running:"
    echo "$remaining_processes"
    echo ""
    echo "🔄 Attempting force kill..."
    echo "$remaining_processes" | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    echo "✅ Force kill completed"
else
    echo "✅ All processes have been successfully terminated"
fi

echo ""
echo "🎉 Process cleanup completed!"
echo "💡 Tip: Use this script whenever you need to stop all project processes"
