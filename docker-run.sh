#!/bin/bash

# =============================
# Docker run script for quotas_forecasting
# This script:
# - Builds the Docker image
# - Retrieves service account key from Secret Manager
# - Runs the container with credentials mounted and passed
# =============================

# Define paths and secret name
TMP_KEY_PATH="/tmp/forecasting-creds.json"         # Where the service account key will be stored temporarily
MOUNT_PATH="/tmp/credentials.json"                 # Path the key will be mounted inside the container
SECRET_NAME="quotas-forecasting-sa-key"            # Name of the secret in Google Secret Manager

# ----------------------------
# Step 1: Build the Docker image
# ----------------------------
echo "🛠️  Building Docker image..."
docker build -t quotas-forecasting .
if [ $? -ne 0 ]; then
  echo "❌ Docker build failed."
  exit 1
fi
echo "✅ Docker image built successfully!"

# ----------------------------
# Step 2: Fetch the service account key from Secret Manager
# ----------------------------
echo "🔐 Fetching service account key from Secret Manager..."
gcloud secrets versions access latest --secret="$SECRET_NAME" > "$TMP_KEY_PATH"

# Check if the secret was successfully retrieved and is not empty
if [ $? -ne 0 ] || [ ! -s "$TMP_KEY_PATH" ]; then
  echo "❌ Failed to retrieve service account key from Secret Manager."
  exit 1
fi

# Make sure the key is readable by Docker (which runs as non-root user)
chmod 644 "$TMP_KEY_PATH"

# ----------------------------
# Step 3: Run the container
# ----------------------------
echo "🚀 Running container with service account credentials..."

# Mount the credentials into the container
# Set the environment variable so GCP SDK can locate the credentials
# Pass all CLI arguments to the container (e.g. --module, --pred-year)
docker run \
  -v "$TMP_KEY_PATH":"$MOUNT_PATH" \
  -e GOOGLE_APPLICATION_CREDENTIALS="$MOUNT_PATH" \
  quotas-forecasting "$@"


# ----------------------------
# Optional cleanup
# Uncomment the line below if you want to delete the temp key after each run
# ----------------------------
# rm "$TMP_KEY_PATH"
