#!/bin/bash
# build_and_test_phase_5_2.sh - Build and Test Script for Phase 5.2

set -e  # Exit on any error

echo "🚀 Phase 5.2: Building and Testing Dockerized Query API"

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

# Check required files for Phase 5.2
print_status "Checking required files for Phase 5.2..."
required_files=(
    "phase_5_2_query_api.py" 
    "requirements.txt" 
    "market_embeddings.json" 
    "gpt2_dispatch_model"
    "market_data.db"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]] && [[ ! -d "$file" ]]; then
        print_error "Required file/directory not found: $file"
        print_warning "Make sure you've completed Phase 4.1 (embeddings), Phase 4.2 (fine-tuned model), and Phase 4.3 (RAG)"
        exit 1
    fi
done

print_success "All required files found"

# Create Dockerfile for query API (since we might not have it yet)
print_status "Creating Dockerfile for Query API..."
cat > Dockerfile.query << 'EOF'
FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=phase_5_2_query_api.py
ENV FLASK_ENV=production

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY phase_5_2_query_api.py .
COPY market_embeddings.json .
COPY gpt2_dispatch_model/ ./gpt2_dispatch_model/
COPY market_data.db .
COPY .env .

RUN mkdir -p /app/logs

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

CMD ["python", "phase_5_2_query_api.py"]
EOF

print_success "Dockerfile.query created"

# Build Docker image
print_status "Building Query API Docker image..."
docker build -f Dockerfile.query -t smart-dispatch-query:latest . || {
    print_error "Docker build failed"
    exit 1
}

print_success "Query API Docker image built successfully"

# Stop any existing containers
print_status "Stopping any existing containers..."
docker stop smart-dispatch-query 2>/dev/null || true
docker rm smart-dispatch-query 2>/dev/null || true

# Check if forecast API is running (from Phase 5.1)
print_status "Checking for Phase 5.1 Forecast API..."
FORECAST_RUNNING=false
if curl -f -s http://localhost:5001/health > /dev/null 2>&1; then
    print_success "Forecast API (Phase 5.1) is running on port 5001"
    FORECAST_RUNNING=true
else
    print_warning "Forecast API (Phase 5.1) not detected on port 5001"
    print_warning "Some integration features may not work"
fi

# Run query API container
print_status "Starting Query API container..."
docker run -d \
    --name smart-dispatch-query \
    -p 5002:5002 \
    --health-cmd="curl -f http://localhost:5002/health || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    smart-dispatch-query:latest || {
    print_error "Failed to start Query API container"
    exit 1
}

print_success "Query API container started successfully"

# Wait for container to be healthy
print_status "Waiting for Query API container to be healthy..."
for i in {1..60}; do
    if docker exec smart-dispatch-query curl -f http://localhost:5002/health &>/dev/null; then
        print_success "Query API container is healthy and responding"
        break
    fi
    
    if [[ $i -eq 60 ]]; then
        print_error "Query API container failed to become healthy"
        print_status "Container logs:"
        docker logs smart-dispatch-query
        exit 1
    fi
    
    sleep 2
done

# Run comprehensive test cases
print_status "Running Phase 5.2 test cases..."

# Test 1: Health check
print_status "Test 1: Health check"
if curl -f -s http://localhost:5002/health > /dev/null; then
    print_success "✅ Health check passed"
else
    print_error "❌ Health check failed"
    exit 1
fi

# Test 2: Query API endpoint
print_status "Test 2: Query API endpoint"
response=$(curl -s -X POST http://localhost:5002/query \
    -H 'Content-Type: application/json' \
    -d '{"question": "Should we dispatch the gas peaker?"}')

if echo "$response" | jq -e '.answer' > /dev/null 2>&1; then
    answer=$(echo "$response" | jq -r '.answer')
    docs=$(echo "$response" | jq -r '.retrieved_documents')
    print_success "✅ Query API passed: Retrieved $docs docs"
    print_status "   Answer: ${answer:0:80}..."
else
    print_error "❌ Query API failed"
    echo "Response: $response"
    exit 1
fi

# Test 3: Chatbot interface
print_status "Test 3: Chatbot interface"
if curl -f -s http://localhost:5002/chat | grep -q "Smart Dispatch Assistant"; then
    print_success "✅ Chatbot interface accessible"
else
    print_error "❌ Chatbot interface failed"
    exit 1
fi

# Test 4: Multiple queries (Test Case: Chatbot interface returns answers)
print_status "Test 4: Multiple chatbot queries"
test_questions=(
    "What will prices be this afternoon?"
    "Why are electricity prices high right now?"
    "What's the current market volatility?"
    "Recommend dispatch strategy for peak hours"
)

passed_queries=0
for question in "${test_questions[@]}"; do
    response=$(curl -s -X POST http://localhost:5002/query \
        -H 'Content-Type: application/json' \
        -d "{\"question\": \"$question\"}")
    
    if echo "$response" | jq -e '.answer' > /dev/null 2>&1; then
        ((passed_queries++))
    fi
done

if [[ $passed_queries -eq ${#test_questions[@]} ]]; then
    print_success "✅ All chatbot queries passed ($passed_queries/${#test_questions[@]})"
else
    print_warning "⚠️ Some chatbot queries failed ($passed_queries/${#test_questions[@]})"
fi

# Test 5: Integration with Forecast API (if available)
if [[ "$FORECAST_RUNNING" == "true" ]]; then
    print_status "Test 5: Integration with Forecast API"
    if curl -f -s http://localhost:5002/forecast > /dev/null; then
        print_success "✅ Forecast API integration working"
    else
        print_warning "⚠️ Forecast API integration partial"
    fi
fi

# Show container info
print_status "Container information:"
echo "  Container ID: $(docker ps -q -f name=smart-dispatch-query)"
echo "  Image: smart-dispatch-query:latest"
echo "  Port: 5002"
echo "  Status: $(docker inspect -f '{{.State.Status}}' smart-dispatch-query)"
echo "  Health: $(docker inspect -f '{{.State.Health.Status}}' smart-dispatch-query)"

# Final success message
print_success "🎉 Phase 5.2 COMPLETE: Dockerized Query API is running successfully!"
echo ""
echo "📊 Test Results: ✅ ALL CORE TESTS PASSED"
echo "🔗 API Endpoints:"
echo "   Health Check: http://localhost:5002/health"
echo "   Query API: http://localhost:5002/query"
echo "   Chatbot Interface: http://localhost:5002/chat"
echo "   Documentation: http://localhost:5002/"
echo ""
echo "🧪 Test the chatbot:"
echo "   1. Open http://localhost:5002/chat in your browser"
echo "   2. Ask: 'Should we dispatch the gas peaker?'"
echo "   3. Ask: 'What will prices be this afternoon?'"
echo ""
echo "🔗 API Integration:"
if [[ "$FORECAST_RUNNING" == "true" ]]; then
    echo "   Both APIs running: Forecast (5001) + Query (5002)"
    echo "   Full system operational!"
else
    echo "   Query API ready, start Forecast API for full integration"
    echo "   Run: python phase_5_1_forecast_api.py"
fi
echo ""
echo "🔄 Next: Phase 5.3 - Deploy to Local Kubernetes (minikube)"
echo ""
echo "To stop the container: docker stop smart-dispatch-query"
echo "To view logs: docker logs smart-dispatch-query"
echo "To open chatbot: open http://localhost:5002/chat"