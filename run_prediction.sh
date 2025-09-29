#!/bin/bash
# Real-time Trading Prediction System Launcher

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

echo -e "${BLUE}🚀 Real-time Crypto Trading Prediction System${NC}"
echo -e "${BLUE}=============================================${NC}\n"

# Function to show usage
show_help() {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 enhanced [OPTIONS]  - Run ENHANCED prediction with live price + BB analysis"
    echo -e "  $0 live [OPTIONS]      - Run LIVE prediction system with real-time data"
    echo -e "  $0 predict [OPTIONS]   - Run prediction system with database data"
    echo -e "  $0 monitor [OPTIONS]   - Run monitoring dashboard"
    echo -e "  $0 both [OPTIONS]      - Run both ENHANCED prediction + monitoring"
    echo -e "  $0 help               - Show this help\n"
    
    echo -e "${YELLOW}Prediction Options:${NC}"
    echo -e "  --timeframe 5m|15m    - Trading timeframe (default: 15m)"
    echo -e "  --model TYPE          - Model type: XGBoost|CatBoost|LightGBM (default: XGBoost)"
    echo -e "  --interval SECONDS    - Update interval (default: 300)"
    echo -e "  --single              - Run single prediction only"
    
    echo -e "${YELLOW}Monitor Options:${NC}"
    echo -e "  --timeframes TF...    - Timeframes to monitor (default: 15m)"
    echo -e "  --interval SECONDS    - Monitor update interval (default: 30)"
    echo -e "  --single              - Run single update only\n"
    
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0 enhanced --timeframe 15m --interval 300  # ENHANCED with BB analysis"
    echo -e "  $0 live --timeframe 15m --interval 300      # LIVE predictions"
    echo -e "  $0 predict --timeframe 15m --interval 300   # Database predictions"
    echo -e "  $0 monitor --timeframes 15m 5m --interval 30"
    echo -e "  $0 both --timeframe 15m                     # ENHANCED + monitoring"
}

# Function to run enhanced prediction system
run_enhanced_prediction() {
    echo -e "${GREEN}🔮 Starting ENHANCED Prediction System...${NC}"
    cd "$PROJECT_ROOT"
    python scripts/predict_enhanced.py "$@"
}

# Function to run live prediction system
run_live_prediction() {
    echo -e "${GREEN}🔮 Starting LIVE Prediction System...${NC}"
    cd "$PROJECT_ROOT"
    python scripts/predict_live.py "$@"
}

# Function to run prediction system
run_prediction() {
    echo -e "${GREEN}🔮 Starting Prediction System...${NC}"
    cd "$PROJECT_ROOT"
    python scripts/predict_simple.py "$@"
}

# Function to run monitoring system
run_monitor() {
    echo -e "${GREEN}🖥️  Starting Monitoring Dashboard...${NC}"
    cd "$PROJECT_ROOT"
    python scripts/monitor_predictions.py "$@"
}

# Function to run both systems
run_both() {
    echo -e "${GREEN}🚀 Starting Both Systems...${NC}"
    echo -e "${YELLOW}Opening monitoring dashboard in new terminal...${NC}"
    
    # Try different terminal commands
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$PROJECT_ROOT' && PYTHONPATH='$PROJECT_ROOT' python scripts/monitor_predictions.py $*; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$PROJECT_ROOT' && PYTHONPATH='$PROJECT_ROOT' python scripts/monitor_predictions.py $*; exec bash" &
    elif command -v konsole &> /dev/null; then
        konsole -e bash -c "cd '$PROJECT_ROOT' && PYTHONPATH='$PROJECT_ROOT' python scripts/monitor_predictions.py $*; exec bash" &
    else
        echo -e "${RED}❌ No supported terminal emulator found. Please run monitoring manually:${NC}"
        echo -e "   python scripts/monitor_predictions.py $*"
    fi
    
    sleep 2
    echo -e "${YELLOW}Starting ENHANCED prediction system in current terminal...${NC}"
    run_enhanced_prediction "$@"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python is not installed or not in PATH${NC}"
    exit 1
fi

# Check if rich is installed
if ! python -c "import rich" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing required package: rich${NC}"
    pip install rich
fi

# Main command handling
case "$1" in
    enhanced)
        shift
        run_enhanced_prediction "$@"
        ;;
    live)
        shift
        run_live_prediction "$@"
        ;;
    predict)
        shift
        run_prediction "$@"
        ;;
    monitor)
        shift
        run_monitor "$@"
        ;;
    both)
        shift
        run_both "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        echo -e "${RED}❌ No command specified${NC}\n"
        show_help
        exit 1
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}\n"
        show_help
        exit 1
        ;;
esac