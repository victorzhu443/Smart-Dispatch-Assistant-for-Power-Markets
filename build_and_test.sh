#!/bin/bash
# build_and_test.sh - Docker Build and Test Script for Phase 5.1

set -e  # Exit on any error

echo "🚀 Phase 5.1: Building and Testing Dockerized Forecast API"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
print_status "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    exit 1
fi

print_success "Docker is running"

# Check required files
print_status "Checking required files..."
required_files=("phase_5_1_forecast_api.py" "Dockerfile" "requirements.txt" "market_data.db")

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "Required file not found: $file"
        exit 1
    fi
done

print_success "All required files found"

# Build Docker image
print_status "Building Docker image..."
docker build -t smart-dispatch-forecast:latest . || {
    print_error "Docker build failed"
    exit 1
}

print_success "Docker image built successfully"

# Stop any existing container
print_status "Stopping any existing containers..."
docker stop smart-dispatch-forecast 2>/dev/null || true
docker rm smart-dispatch-forecast 2>/dev/null || true

# Run container
print_status "Starting container..."
docker run -d \
    --name smart-dispatch-forecast \
    -p 5001:5001 \
    --health-cmd="curl -f http://localhost:5001/health || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    smart-dispatch-forecast:latest || {
    print_error "Failed to start container"
    exit 1
}

print_success "Container started successfully"

# Wait for container to be healthy
print_status "Waiting for container to be healthy..."
for i in {1..30}; do
    if docker exec smart-dispatch-forecast curl -f http://localhost:5001/health &>/dev/null; then
        print_success "Container is healthy and responding"
        break
    fi
    
    if [[ $i -eq 30 ]]; then
        print_error "Container failed to become healthy"
        print_status "Container logs:"
        docker logs smart-dispatch-forecast
        exit 1
    fi
    
    sleep 2
done

# Run test cases
print_status "Running API test cases..."

# Test 1: Health check
print_status "Test 1: Health check"
if curl -f -s http://localhost:5001/health > /dev/null; then
    print_success "✅ Health check passed"
else
    print_error "❌ Health check failed"
    exit 1
fi

# Test 2: Default forecast
print_status "Test 2: Default forecast"
response=$(curl -s http://localhost:5001/forecast)
if echo "$response" | jq -e '.predicted_price' > /dev/null 2>&1; then
    price=$(echo "$response" | jq -r '.predicted_price')
    print_success "✅ Default forecast passed: \$${price}/MWh"
else
    print_error "❌ Default forecast failed"
    echo "Response: $response"
    exit 1
fi

# Test 3: Specific timestamp forecast
print_status "Test 3: Specific timestamp forecast"
response=$(curl -s "http://localhost:5001/forecast?timestamp=2025-07-31T15:30:00")
if echo "$response" | jq -e '.predicted_price' > /dev/null 2>&1; then
    price=$(echo "$response" | jq -r '.predicted_price')
    timestamp=$(echo "$response" | jq -r '.timestamp')
    print_success "✅ Timestamp forecast passed: \$${price}/MWh for ${timestamp}"
else
    print_error "❌ Timestamp forecast failed"
    echo "Response: $response"
    exit 1
fi

# Test 4: API documentation
print_status "Test 4: API documentation"
if curl -f -s http://localhost:5001/ > /dev/null; then
    print_success "✅ API documentation accessible"
else
    print_error "❌ API documentation failed"
    exit 1
fi

# Show container info
print_status "Container information:"
echo "  Container ID: $(docker ps -q -f name=smart-dispatch-forecast)"
echo "  Image: smart-dispatch-forecast:latest"
echo "  Port: 5001"
echo "  Status: $(docker inspect -f '{{.State.Status}}' smart-dispatch-forecast)"
echo "  Health: $(docker inspect -f '{{.State.Health.Status}}' smart-dispatch-forecast)"

# Final success message
print_success "🎉 Phase 5.1 COMPLETE: Dockerized Forecast API is running successfully!"
echo ""
echo "📊 Test Results: ✅ ALL PASSED"
echo "🔗 API Endpoints:"
echo "   Health Check: http://localhost:5001/health"
echo "   Forecast API: http://localhost:5001/forecast"
echo "   Documentation: http://localhost:5001/"
echo ""
echo "🧪 Test the API:"
echo "   curl http://localhost:5001/forecast"
echo "   curl \"http://localhost:5001/forecast?timestamp=2025-07-31T15:30:00\""
echo ""
echo "🔄 Next: Phase 5.2 - Dockerize /query API"
echo ""
echo "To stop the container: docker stop smart-dispatch-forecast"
echo "To view logs: docker logs smart-dispatch-forecast"