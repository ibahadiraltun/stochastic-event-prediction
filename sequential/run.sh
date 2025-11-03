#!/bin/bash

# STEP Sequential - Convenience Run Script
# Usage: ./run.sh [command] [args...]

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "STEP Sequential - Run Script"
    echo ""
    echo "Usage: ./run.sh [command] [args...]"
    echo ""
    echo "Commands:"
    echo "  build              Build the project"
    echo "  clean              Clean build artifacts"
    echo "  rebuild            Clean and rebuild"
    echo "  predict [args]     Run event prediction"
    echo "  score [args]       Run evaluation scorer"
    echo "  example            Run example with test data"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh build"
    echo "  ./run.sh predict data/network.txt output.txt 0.8 100"
    echo "  ./run.sh score data/network.txt output.txt 0.8 0.2"
}

build_project() {
    echo -e "${GREEN}Building STEP Sequential...${NC}"

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # Use new CMakeLists if available, otherwise fall back to original
    if [ -f "${PROJECT_DIR}/CMakeLists_new.txt" ]; then
        echo -e "${YELLOW}Using refactored CMakeLists...${NC}"
        cp "${PROJECT_DIR}/CMakeLists_new.txt" "${PROJECT_DIR}/CMakeLists.txt"
    fi

    cmake ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    echo -e "${GREEN}Build complete!${NC}"
    echo -e "Executables:"
    echo -e "  - ${BUILD_DIR}/step-sequential"
    echo -e "  - ${BUILD_DIR}/scorer"
}

clean_project() {
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    rm -rf "${BUILD_DIR}"
    echo -e "${GREEN}Clean complete!${NC}"
}

rebuild_project() {
    clean_project
    build_project
}

run_prediction() {
    if [ ! -f "${BUILD_DIR}/step-sequential" ]; then
        echo -e "${RED}Error: step-sequential not built. Run './run.sh build' first.${NC}"
        exit 1
    fi

    if [ $# -lt 4 ]; then
        echo -e "${RED}Error: Insufficient arguments for prediction${NC}"
        echo "Usage: ./run.sh predict <input_file> <output_file> <train_ratio> <prediction_size>"
        exit 1
    fi

    echo -e "${GREEN}Running STEP prediction...${NC}"
    "${BUILD_DIR}/step-sequential" "$@"
}

run_scorer() {
    if [ ! -f "${BUILD_DIR}/scorer" ]; then
        echo -e "${RED}Error: scorer not built. Run './run.sh build' first.${NC}"
        exit 1
    fi

    if [ $# -lt 4 ]; then
        echo -e "${RED}Error: Insufficient arguments for scoring${NC}"
        echo "Usage: ./run.sh score <ground_truth> <predictions> <train_ratio> <test_ratio>"
        exit 1
    fi

    echo -e "${GREEN}Running scorer...${NC}"
    "${BUILD_DIR}/scorer" "$@"
}

run_example() {
    echo -e "${GREEN}Running example workflow...${NC}"

    # Check if build exists
    if [ ! -f "${BUILD_DIR}/step-sequential" ]; then
        echo -e "${YELLOW}Building project first...${NC}"
        build_project
    fi

    # Create example data if it doesn't exist
    EXAMPLE_DATA="${PROJECT_DIR}/example_data.txt"
    if [ ! -f "${EXAMPLE_DATA}" ]; then
        echo -e "${YELLOW}Generating example data...${NC}"
        cat > "${EXAMPLE_DATA}" << 'EOF'
1 2 1000
2 3 1050
1 3 1100
3 4 1150
4 5 1200
1 4 1250
2 4 1300
3 5 1350
1 5 1400
2 5 1450
EOF
        echo -e "${GREEN}Example data created at ${EXAMPLE_DATA}${NC}"
    fi

    EXAMPLE_OUTPUT="${PROJECT_DIR}/example_predictions.txt"

    echo -e "${YELLOW}Running prediction on example data...${NC}"
    run_prediction "${EXAMPLE_DATA}" "${EXAMPLE_OUTPUT}" 0.7 5

    echo ""
    echo -e "${YELLOW}Running evaluation...${NC}"
    run_scorer "${EXAMPLE_DATA}" "${EXAMPLE_OUTPUT}" 0.7 0.3

    echo ""
    echo -e "${GREEN}Example complete!${NC}"
    echo -e "Predictions saved to: ${EXAMPLE_OUTPUT}"
}

# Main script logic
case "${1:-help}" in
    build)
        build_project
        ;;
    clean)
        clean_project
        ;;
    rebuild)
        rebuild_project
        ;;
    predict)
        shift
        run_prediction "$@"
        ;;
    score)
        shift
        run_scorer "$@"
        ;;
    example)
        run_example
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
