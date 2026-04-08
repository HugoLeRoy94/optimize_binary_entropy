# 1. Start with the official NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:24.01-py3

# 2. Create a non-root user matching your host UID (1000)
RUN groupadd -g 1000 devuser && \
    useradd -u 1000 -g 1000 -m -s /bin/bash devuser

# 2. Set the working directory inside the container
WORKDIR /app

# Copy requirements file from your host to the container
COPY requirements.txt .

# Install the listed modules
RUN pip install --no-cache-dir -r requirements.txt

# 4. Expose the port Jupyter will run on
EXPOSE 8888

# 5. The command to start Jupyter when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='docker'"]
