#!/bin/bash
# =============================================================================
# Batch Environment Builder for Skills
# =============================================================================
# This script builds conda environments based on skill_env_mapping.json
#
# Usage:
#   ./build_envs.sh [mapping_file] [--dry-run]
#
# Arguments:
#   mapping_file  Path to skill_env_mapping.json (default: ./skill_env_mapping.json)
#   --dry-run     Print commands without executing
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MAPPING_FILE="${1:-skill_env_mapping.json}"
DRY_RUN=false
CONDA_BASE="/opt/conda"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
    esac
done

# Check if mapping file exists
if [ ! -f "$MAPPING_FILE" ]; then
    echo -e "${RED}Error: Mapping file not found: $MAPPING_FILE${NC}"
    echo "Run: python -m harbor_synthesis.env_builder.main --output skill_env_mapping.json"
    exit 1
fi

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed${NC}"
    echo "Install with: apt-get install jq"
    exit 1
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    Batch Environment Builder for Skills    ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Mapping file: $MAPPING_FILE"
echo "Dry run: $DRY_RUN"
echo ""

# Get list of environments
ENVS=$(jq -r '.environments | keys[]' "$MAPPING_FILE")
TOTAL_ENVS=$(echo "$ENVS" | wc -l)

echo -e "${GREEN}Found $TOTAL_ENVS environments to build${NC}"
echo ""

# Function to run or print command
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

# Build each environment
count=0
for ENV_NAME in $ENVS; do
    count=$((count + 1))

    echo -e "${YELLOW}============================================${NC}"
    echo -e "${YELLOW}[$count/$TOTAL_ENVS] Building environment: $ENV_NAME${NC}"
    echo -e "${YELLOW}============================================${NC}"

    # Get environment config
    PYTHON_VERSION=$(jq -r ".environments.\"$ENV_NAME\".python_version // \"3.10\"" "$MAPPING_FILE")
    DESCRIPTION=$(jq -r ".environments.\"$ENV_NAME\".description // \"\"" "$MAPPING_FILE")
    SKILLS=$(jq -r ".environments.\"$ENV_NAME\".skills | join(\", \")" "$MAPPING_FILE")

    echo "Description: $DESCRIPTION"
    echo "Python: $PYTHON_VERSION"
    echo "Skills: $SKILLS"
    echo ""

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Environment $ENV_NAME already exists, will update${NC}"
    else
        # Create new environment
        echo -e "${GREEN}Creating environment: $ENV_NAME${NC}"
        run_cmd conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    fi

    # Get conda packages
    CONDA_PKGS=$(jq -r ".environments.\"$ENV_NAME\".conda_packages | join(\" \")" "$MAPPING_FILE")
    if [ -n "$CONDA_PKGS" ] && [ "$CONDA_PKGS" != "null" ]; then
        echo -e "${GREEN}Installing conda packages: $CONDA_PKGS${NC}"
        run_cmd conda install -n "$ENV_NAME" -c conda-forge $CONDA_PKGS -y
    fi

    # Get pip packages
    PIP_PKGS=$(jq -r ".environments.\"$ENV_NAME\".pip_packages | join(\" \")" "$MAPPING_FILE")
    if [ -n "$PIP_PKGS" ] && [ "$PIP_PKGS" != "null" ]; then
        echo -e "${GREEN}Installing pip packages...${NC}"
        # Use conda run to ensure correct environment
        run_cmd conda run -n "$ENV_NAME" pip install $PIP_PKGS
    fi

    # Verify installation
    echo -e "${GREEN}Verifying environment...${NC}"
    if [ "$DRY_RUN" = false ]; then
        conda run -n "$ENV_NAME" python --version
        echo -e "${GREEN}Environment $ENV_NAME built successfully${NC}"
    fi

    echo ""
done

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    Build Complete!                         ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Environments created: $TOTAL_ENVS"
echo ""
echo "To use an environment:"
echo "  conda activate <env_name>"
echo "  # or"
echo "  conda run -n <env_name> python script.py"
echo ""
echo "To list all environments:"
echo "  conda env list"
