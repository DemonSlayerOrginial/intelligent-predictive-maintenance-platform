# Kubernetes deployment

The application tier is Kubernetes-ready in `apps.yaml`. In production, run Kafka,
PostgreSQL, and Redis as managed/stateful infrastructure (or install your preferred
operators/Helm charts) and make their Kubernetes service names match `pmp-config`.

1. Build and push the root Docker image.
2. Replace `YOUR_REGISTRY/predictive-maintenance:latest`.
3. Replace the demonstration database secret.
4. Mount the trained `models/` bundle into the inference pods or bake a versioned
   model bundle into the image/object-store init container.
5. Apply with `kubectl apply -f k8s/apps.yaml`.

The inference deployment includes an HPA. In a production version, scale stream
processors by Kafka lag rather than CPU alone (for example with KEDA).
