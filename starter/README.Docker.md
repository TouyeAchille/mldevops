---
noteId: "f97ab3b0713c11f0b4480d6fe6b04a1e"
tags: []

---

### Building and running your application

When you're ready, start your application by running:
`docker compose up --build`.

Your application will be available at http://localhost:8080.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)

Sure! Here's a concise explanation of your `Dockerfile` written in English and formatted in Markdown:

###  Dockerfile Explanation

This `Dockerfile` is designed to build a lightweight and secure container image for serving a FastAPI application used for census income prediction.

It starts from the official slim Python base image (`python:3.11.0-slim`) to reduce image size and improve performance. Two environment variables are set: `PYTHONDONTWRITEBYTECODE=1` disables Python bytecode generation, and `PYTHONUNBUFFERED=1` ensures that logs are immediately flushed to the console, which is important for debugging and monitoring.

A non-root user (`appuser`) is created using a fixed UID (`10001`) with no home directory and no login shell, following Docker security best practices. This prevents the application from running as root inside the container, reducing the risk of security vulnerabilities.

The `RUN` instruction uses Docker BuildKit features to optimize dependency installation. It leverages a **cache mount** for `pip` to speed up repeated builds and a **bind mount** for `requirements.txt` to avoid adding it to the image layer manually. This approach keeps the image clean and enables efficient rebuilds when dependencies change.

After installing the dependencies, the application source code is copied into the container. The container switches to the non-privileged `appuser` and exposes port `8080`, where the FastAPI app will be served using `uvicorn`.

This setup ensures a production-friendly container image that is small, secure, and efficient to build and run.
