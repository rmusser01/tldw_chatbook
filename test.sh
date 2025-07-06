#!/bin/bash
# Quick test runner wrapper for common scenarios

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default to quick mode
MODE="quick"

# Parse arguments
case "$1" in
    "smoke")
        echo -e "${BLUE}Running smoke tests...${NC}"
        python run_tests.py --mode smoke
        ;;
    "quick")
        echo -e "${BLUE}Running quick tests...${NC}"
        python run_tests.py --mode quick
        ;;
    "full")
        echo -e "${BLUE}Running full test suite...${NC}"
        python run_tests.py --mode full
        ;;
    "coverage")
        echo -e "${BLUE}Running tests with coverage...${NC}"
        python run_tests.py --mode coverage
        ;;
    "parallel")
        echo -e "${BLUE}Running tests in parallel...${NC}"
        python run_tests.py --mode full --parallel
        ;;
    "failed")
        echo -e "${BLUE}Re-running previously failed tests...${NC}"
        pytest --lf -v
        ;;
    "changed")
        echo -e "${BLUE}Running tests for changed files...${NC}"
        # Get changed Python files
        CHANGED_FILES=$(git diff --name-only HEAD | grep -E '\.py$' || true)
        if [ -z "$CHANGED_FILES" ]; then
            echo -e "${YELLOW}No Python files changed${NC}"
            exit 0
        fi
        # Find corresponding test files
        TEST_FILES=""
        for file in $CHANGED_FILES; do
            # Convert source file to test file path
            test_file=$(echo "$file" | sed 's|tldw_chatbook/|Tests/|' | sed 's|\.py$|_test.py|')
            if [ -f "$test_file" ]; then
                TEST_FILES="$TEST_FILES $test_file"
            fi
            # Also check for test_ prefix
            test_file=$(echo "$file" | sed 's|tldw_chatbook/|Tests/|' | sed 's|/\([^/]*\)\.py$|/test_\1.py|')
            if [ -f "$test_file" ]; then
                TEST_FILES="$TEST_FILES $test_file"
            fi
        done
        if [ -z "$TEST_FILES" ]; then
            echo -e "${YELLOW}No test files found for changed files${NC}"
            exit 0
        fi
        echo -e "${GREEN}Running tests: $TEST_FILES${NC}"
        pytest $TEST_FILES -v
        ;;
    "group")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please specify a test group${NC}"
            echo "Available groups: database, core, integration, ui, utility"
            exit 1
        fi
        echo -e "${BLUE}Running $2 tests...${NC}"
        python run_tests.py --groups "$2"
        ;;
    "watch")
        echo -e "${BLUE}Starting test watcher...${NC}"
        if ! command -v ptw &> /dev/null; then
            echo -e "${YELLOW}Installing pytest-watch...${NC}"
            pip install pytest-watch
        fi
        ptw -- -v --tb=short
        ;;
    "clean")
        echo -e "${BLUE}Cleaning test artifacts...${NC}"
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        find . -type f -name ".coverage" -delete 2>/dev/null || true
        echo -e "${GREEN}Test artifacts cleaned${NC}"
        ;;
    "report")
        # Open the latest HTML report
        LATEST_REPORT=$(ls -t test_reports/*/report.html 2>/dev/null | head -1)
        if [ -z "$LATEST_REPORT" ]; then
            echo -e "${RED}No test reports found${NC}"
            exit 1
        fi
        echo -e "${BLUE}Opening latest test report...${NC}"
        if command -v open &> /dev/null; then
            open "$LATEST_REPORT"
        elif command -v xdg-open &> /dev/null; then
            xdg-open "$LATEST_REPORT"
        else
            echo -e "${GREEN}Report: $LATEST_REPORT${NC}"
        fi
        ;;
    "help"|"--help"|"-h"|"")
        echo "Test Runner for tldw_chatbook"
        echo ""
        echo "Usage: ./test.sh [command]"
        echo ""
        echo "Commands:"
        echo "  smoke      Run minimal smoke tests (fastest)"
        echo "  quick      Run quick tests - critical groups only (default)"
        echo "  full       Run full test suite"
        echo "  coverage   Run tests with coverage report"
        echo "  parallel   Run full tests in parallel"
        echo "  failed     Re-run previously failed tests"
        echo "  changed    Run tests for changed files only"
        echo "  group <g>  Run specific test group"
        echo "  watch      Start test watcher (auto-run on changes)"
        echo "  clean      Clean test artifacts and caches"
        echo "  report     Open latest HTML test report"
        echo "  help       Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./test.sh              # Run quick tests"
        echo "  ./test.sh full         # Run all tests"
        echo "  ./test.sh group core   # Run core tests only"
        echo "  ./test.sh changed      # Test changed files"
        echo ""
        echo "Available test groups:"
        echo "  database    Database layer tests"
        echo "  core        Core feature tests"
        echo "  integration Integration tests"
        echo "  ui          UI and widget tests"
        echo "  utility     Utility and infrastructure tests"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Run './test.sh help' for usage information"
        exit 1
        ;;
esac