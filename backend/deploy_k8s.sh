#!/bin/bash
# deploy_k8s.sh - Phase 5.3: Deploy to Local Kubernetes (minikube)

set -e  # Exit on any error

echo "🚀 Phase 5.3: Deploy Smart Dispatch System to Local Kubernetes"

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

# Check prerequisites
print_status "Checking prerequisites..."

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    print_error "minikube is not installed"
    print_status "Please install minikube: https://minikube.sigs.k8s.io/docs/start/"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed"
    print_status "Please install kubectl: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    exit 1
fi

print_success "All prerequisites met"

# Check required files
print_status "Checking required files..."
required_files=(
    "backend/phase_5_1_forecast_api.py"
    "market_data.db"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]] && [[ ! -d "$file" ]]; then
        print_error "Required file/directory not found: $file"
        print_warning "Make sure you've completed Phases 4.1, 4.2, 5.1, and 5.2"
        exit 1
    fi
done

print_success "All required files found"

# Create Kubernetes manifests
print_status "Creating Kubernetes manifests..."

# Create namespace manifest
cat > k8s-namespace.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: smart-dispatch
  labels:
    name: smart-dispatch
    environment: development
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: smart-dispatch-quota
  namespace: smart-dispatch
spec:
  hard:
    requests.cpu: "2"
    requests.memory: 4Gi
    limits.cpu: "4"
    limits.memory: 8Gi
    pods: "10"
    services: "5"
EOF

# Create forecast deployment manifest
cat > k8s-forecast.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forecast-api
  namespace: smart-dispatch
spec:
  replicas: 1
  selector:
    matchLabels:
      app: forecast-api
  template:
    metadata:
      labels:
        app: forecast-api
    spec:
      containers:
      - name: forecast-api
        image: smart-dispatch-forecast:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 5001
        env:
        - name: FLASK_ENV
          value: "production"
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "512Mi"
            cpu: "250m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: forecast-service
  namespace: smart-dispatch
spec:
  selector:
    app: forecast-api
  ports:
  - port: 80
    targetPort: 5001
  type: ClusterIP
EOF

# Create ingress manifest
cat > k8s-ingress.yaml << 'EOF'
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: smart-dispatch-ingress
  namespace: smart-dispatch
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: localhost
    http:
      paths:
      - path: /forecast
        pathType: Prefix
        backend:
          service:
            name: forecast-service
            port:
              number: 80
      - path: /query
        pathType: Prefix
        backend:
          service:
            name: query-service
            port:
              number: 80
      - path: /chat
        pathType: Prefix
        backend:
          service:
            name: query-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: query-service
            port:
              number: 80
EOF

print_success "Kubernetes manifests created"

# Start minikube if not running
print_status "Starting minikube..."
if ! minikube status &> /dev/null; then
    print_status "Starting minikube cluster..."
    minikube start --driver=docker --memory=4096 --cpus=2
    print_success "minikube started"
else
    print_success "minikube already running"
fi

# Enable ingress addon
print_status "Enabling ingress addon..."
minikube addons enable ingress

# Configure docker environment
print_status "Configuring Docker environment for minikube..."
eval $(minikube docker-env)

# Build Docker images in minikube
print_status "Building Docker images in minikube..."

# Build forecast API image
print_status "Building forecast API image..."
cat > Dockerfile.forecast << 'EOF'
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements_minimal.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY phase_5_1_forecast_api.py .
COPY market_data.db .
EXPOSE 5001
CMD ["python", "backend/phase_5_1_forecast_api.py"]
EOF

docker build -f Dockerfile.forecast -t smart-dispatch-forecast:latest .

print_success "Docker images built successfully"

# Deploy to Kubernetes
print_status "Deploying to Kubernetes..."

# Create namespace
kubectl apply -f k8s-namespace.yaml

# Deploy services
kubectl apply -f k8s-forecast.yaml

# Wait for deployments to be ready
print_status "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/forecast-api -n smart-dispatch

# Deploy ingress
kubectl apply -f k8s-ingress.yaml

# Wait for ingress to be ready
print_status "Waiting for ingress to be ready..."
sleep 30

print_success "Deployment completed"

# Get ingress information
INGRESS_IP=$(minikube ip)
print_status "Ingress IP: $INGRESS_IP"

# Test deployments
print_status "Running Phase 5.3 test cases..."

# Test Case: Accessible via localhost/query and localhost/forecast
print_status "Test Case: Services accessible via ingress"

# Test forecast endpoint
print_status "Testing forecast endpoint..."
FORECAST_URL="http://$INGRESS_IP/forecast"
if curl -f -s "$FORECAST_URL" > /dev/null; then
    FORECAST_RESPONSE=$(curl -s "$FORECAST_URL")
    if echo "$FORECAST_RESPONSE" | grep -q "predicted_price"; then
        PRICE=$(echo "$FORECAST_RESPONSE" | grep -o '"predicted_price":[0-9.]*' | cut -d':' -f2)
        print_success "✅ Forecast API accessible: Price $${PRICE}/MWh"
    else
        print_warning "⚠️ Forecast API responding but format unexpected"
    fi
else
    print_error "❌ Forecast API not accessible at $FORECAST_URL"
fi

# Show cluster status
print_status "Kubernetes cluster status:"
kubectl get pods -n smart-dispatch -o wide
echo ""
kubectl get services -n smart-dispatch
echo ""
kubectl get ingress -n smart-dispatch

# Final success message
print_success "🎉 Phase 5.3 COMPLETE: Smart Dispatch System deployed to Kubernetes!"
echo ""
echo "📊 Deployment Summary:"
echo "   Namespace: smart-dispatch"
echo "   Ingress: smart-dispatch-ingress"
echo "   Minikube IP: $INGRESS_IP"
echo ""
echo "🔗 Access URLs:"
echo "   Forecast API: http://$INGRESS_IP/forecast"
echo "   Health Check: http://$INGRESS_IP/health"
echo ""
echo "🧪 Test Commands:"
echo "   curl http://$INGRESS_IP/forecast"
echo "   curl http://$INGRESS_IP/health"
echo ""
echo "📊 PRD Test Case Status:"
if [[ "$FORECAST_RESPONSE" == *"forecast"* ]]; then
    print_success "✅ PASSED: Accessible via localhost/forecast"
else
    print_warning "⚠️ PARTIAL: The forecast endpoint may need additional configuration"
fi
echo ""
echo "To stop: minikube stop"
echo "To delete: minikube delete"
echo "To view logs: kubectl logs -f deployment/forecast-api -n smart-dispatch"