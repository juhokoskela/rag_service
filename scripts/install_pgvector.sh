#!/bin/bash
set -e

echo "Installing pgvector extension..."

# Switch to root to install packages
if [ "$EUID" -ne 0 ]; then
    echo "Script needs to run as root for package installation"
    exit 1
fi

# Create apt directories if they don't exist
mkdir -p /var/lib/apt/lists/partial
chmod 755 /var/lib/apt/lists/partial

# Update package list
apt-get update

# Install build dependencies
apt-get install -y \
    git \
    build-essential \
    postgresql-server-dev-17 \
    wget \
    ca-certificates

# Clone and build pgvector
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector

# Use latest stable version
git checkout v0.8.0

# Build and install
make clean
make OPTFLAGS=""
make install

echo "pgvector extension installed successfully"

# Clean up
cd /
rm -rf /tmp/pgvector
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "pgvector installation complete"
