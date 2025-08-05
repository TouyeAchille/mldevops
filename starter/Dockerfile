ARG PYTHON_VERSION=3.11.0
FROM python:${PYTHON_VERSION}-slim AS base

LABEL maintainer="Achille Mbogol <touyejunior@gmail.com>"
# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

WORKDIR /app

# Copy the source code into the container.
COPY . .
# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
#--mount=type=cache,target=/root/.cache/pip \
#--mount=type=bind,source=requirements.txt,target=requirements.txt \
#python -m pip install -r requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Switch to the non-privileged user to run the app lication.
USER appuser


# Expose the port that the application listens on.
EXPOSE 8080

# Run the application.
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
