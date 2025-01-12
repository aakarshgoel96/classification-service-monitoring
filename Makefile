.PHONY: all build deploy prometheus grafana clean

build:
    docker build -t classification-inference-service:latest -f apps/classification-inference-service/Dockerfile apps/classification-inference-service/
    docker build -t streamlit-app:latest -f apps/streamlit-app/Dockerfile apps/streamlit-app/

deploy:
    kubectl apply -f namespace.yaml
    kubectl apply -f k8s/classification-inference-service/   --namespace=classification-inference
    kubectl apply -f k8s/streamlit-app/    --namespace=classification-inference

prometheus:
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    helm install prometheus prometheus-community/prometheus
    kubectl apply -f k8s/prometheus/configmap.yaml -n default

grafana:
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    helm install grafana grafana/grafana

clean:
    kubectl delete namespace classification-inference
    helm uninstall prometheus
    helm uninstall grafana
