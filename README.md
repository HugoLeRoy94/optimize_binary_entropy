# 🐳 Docker & DevPod Cheat Sheet

## 1. Concepts
* **Docker:** The "Box." It creates an isolated, virtual computer (container) inside your machine that has all the specific software (Python, CUDA, PyTorch) your project needs. It ensures your code runs exactly the same everywhere.
* **DevPod:** The "Bridge." It connects your local IDE (VSCodium) to the Docker container. It handles the complex SSH tunneling and file syncing so you can edit code inside the container as if it were on your own hard drive.

---

## 2. Key Files
| File | Role | Metaphor |
| :--- | :--- | :--- |
| **`Dockerfile`** | **The Recipe.** Tells Docker how to build the OS. "Start with NVIDIA Linux, add Python 3.10, install these libraries." | The Blueprint |
| **`docker-compose.yaml`** | **The Orchestrator.** Tells Docker how to run the container. "Use the GPU, map port 8888, sync the local folder to `/app`, and set shared memory to 2GB." | The Director |
| **`.devcontainer/devcontainer.json`** | **The Config for DevPod.** Tells DevPod which `docker-compose` file to use and which IDE extensions (Python, Jupyter) to install inside the container automatically. | The Connection Settings |

---

## 3. Command Line Survival Guide

### 🐳 Docker Commands (The Engine)
* **Start Everything:**
    ```bash
    sudo docker-compose up       # Starts the container and shows logs
    sudo docker-compose up -d    # Starts in "detached" mode (background)
    sudo docker-compose up --build # Force rebuilds the image (use if you changed requirements.txt)
    ```

* **Check Status:**
    ```bash
    sudo docker ps               # List currently running containers
    sudo docker ps -a            # List ALL containers (including stopped ones)
    ```

* **Stop Everything:**
    ```bash
    sudo docker-compose stop     # Pauses the containers (keeps state)
    sudo docker-compose down     # Stops and removes containers/networks (Clean slate)
    docker-compose up -d         # update the container, if I just changed the docker-compose file
    ```

* **Kill/Remove Containers (Fix Conflicts):**
    ```bash
    sudo docker rm -f <container_name>  # Force remove a specific container
    sudo docker rm -f $(sudo docker ps -aq)  # Nuke option: Remove ALL containers
    ```

* **View Logs:**
    ```bash
    sudo docker logs -f <container_name> # Follow the logs in real-time
    ```

### 😈 DevPod Commands (The IDE Bridge)
* **Start Work (The Magic Command):**
    ```bash
    devpod up .                 # Reads .devcontainer.json and launches Codium
    ```

* **Specify IDE (If it opens browser):**
    ```bash
    devpod up . --ide codium    # Forces VSCodium to open
    devpod up . --recreate      # Remake the workspace, if docker-compose has been changed for instance
    ```

* **Manage Workspaces:**
    ```bash
    devpod list                 # See active workspaces
    devpod stop <name>          # Stop a workspace to save GPU/RAM
    devpod delete <name>        # Delete the workspace setup (Files remain safe)
    ```

* **Fix IDE Path (Fedora Specific):**
    ```bash
    devpod ide set-options codium -o COMMAND=$(which codium)
    ```

# Run a simple Python Script
```
docker exec -it optimize_binary python3 your_script.py
```

-it is just to keep it interactive

# 🚀 GPU Cluster Survival Guide (e4-seminara)

This guide summarizes the final working configuration for running code on the University A100 GPUs using **Docker** while bypassing the broken system Python and complex SSH tunnels.

---

## 1. The Setup Architecture
We are using **Rootless Docker** to manage dependencies and GPU access without needing `sudo` permissions.

* **Host Server:** `leroy@10.187.172.7`
* **Container Name:** `optimize_binary`
* **GPU Hardware:** 4x NVIDIA A100
* **Mapping:** Your local folder `/home/leroy/optimize_binary_entropy` is mirrored inside the container at `/app`.

---

## 2. How to Work (The Workflow)
To avoid the "SSH Inception" and connection errors, we use VSCodium to edit files on the host and the terminal to execute them in the container.

### Step 1: Connect to the Server
1. Open **VSCodium** on your laptop.
2. Use the **Remote - SSH** extension to connect to `leroy@10.187.172.7`.
3. Open your project folder: `/home/leroy/optimize_binary_entropy`.
4. **Edit your code here.** Saving a file here automatically updates it inside the Docker container.

### Step 2: Run Your Code (The Docker "Bridge")
Do not run `python` directly in the VSCodium terminal. Instead, "inject" your commands into the running Docker container using `docker exec`.

* **To check GPU availability:**
    ```bash
    docker exec -it optimize_binary nvidia-smi
    ```
* **To run a Python script:**
    ```bash
    docker exec -it optimize_binary python3 your_script.py
    ```
* **To get an interactive shell inside the GPU environment:**
    ```bash
    docker exec -it optimize_binary bash
    ```

---

## 3. Important Configurations

### University Proxy (Inside the Container)
If your code needs to download data or packages, run these exports inside the container terminal:
```bash
export http_proxy=[http://proxy.unige.it:8080/](http://proxy.unige.it:8080/)
export https_proxy=[http://proxy.unige.it:8080/](http://proxy.unige.it:8080/)
