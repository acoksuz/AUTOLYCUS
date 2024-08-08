# Use the official Jupyter base image
FROM jupyter/base-notebook:latest

USER root
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install any dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Clone your GitHub repository
RUN git clone https://github.com/acoksuz/AUTOLYCUS.git /home/jovyan/AUTOLYCUS

# Change working directory to the project directory
WORKDIR /home/jovyan/AUTOLYCUS

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
