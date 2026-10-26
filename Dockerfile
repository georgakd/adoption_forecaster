# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set general environment settings:
## Prevents creation of .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

## Forces stdout/stderr to be flushed immediately (for logging)       
ENV PYTHONUNBUFFERED=1

## Avoids prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive   

## Enables BuildKit for better Docker builds
ENV DOCKER_BUILDKIT=1               

# Project-level identifiers
ENV GOOGLE_CLOUD_PROJECT=YOUR_GCP_PROJECT_ID
ENV GCS_BUCKET_NAME=adoption-forecaster-data
# Add /app to Python path so adoption_forecaster package can be found
ENV PYTHONPATH=/app

# Install core packages:
# - curl: for downloading Poetry installer
# - build-essential: compilers and headers for Python packages
# - git: in case any dependencies are from git repos
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*


# Install Poetry (Python dependency manager) and link it to global path
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set the working directory inside the image
WORKDIR /app

# Copy dependency definition files (used to install Python packages)
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
# We disable Poetry's virtualenvs and install into the system-wide Python inside the container
RUN poetry config virtualenvs.create false && poetry install --only=main --no-root

# Copy the actual forecasting source code into the image
COPY adoption_forecaster/ ./adoption_forecaster/

# Create a non-root user for running the app securely
# Running as root inside containers is a bad practice — it can be a security risk, especially if this image is ever used in production or exposed externally.
# Here, we manually create a group and a user with a fixed UID and GID.
# 8002 is just an arbitrary number (we used in SIA, just used the same for consistency), but it could be anything ≥100.
# (UID/GID values 1-99 are generally reserved for system use)
RUN addgroup --gid 8002 appgroup && \
    adduser --uid 8002 --ingroup appgroup --disabled-password appuser

# Run all remaining commands as this secure, non-root user
USER appuser

# Set the default command to run when the container starts.
# This means: when you run `docker run adoption-forecaster --args...`
# ...it will actually run: `python adoption_forecaster/main.py --args...`
ENTRYPOINT ["python", "adoption_forecaster/main.py"]
