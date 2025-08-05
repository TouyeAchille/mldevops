ARG PYTHON_VERSION=3.11.0
FROM python:${PYTHON_VERSION}-slim AS base

LABEL maintainer="Achille Mbogol <touyejunior@gmail.com>"
# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
ARG UID=10001

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

WORKDIR /app

# Copy the source code into the container.
COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Expose the port that the application listens on.
EXPOSE 8080

# Switch to the non-privileged user to run the app lication.
USER appuser

# Run the application.
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
