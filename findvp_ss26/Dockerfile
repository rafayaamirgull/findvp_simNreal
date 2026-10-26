FROM ubuntu:24.04

USER root
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    cmake-curses-gui \
    curl \
    gdb \
    git \
    g++ \
    gcc \
    htop \
    libasound2t64 \
    libboost-all-dev \
    libceres-dev \
    libeigen3-dev \
    libflann-dev \
    libgl1 \
    libglib2.0-0 \
    libopencv-dev \
    libpcl-dev \
    libpulse0 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libyaml-cpp-dev \
    locales \
    make \
    mesa-utils \
    nano \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-opencv \
    python3-pil \
    python3-pip \
    python3-scipy \
    python3-skimage \
    python3-sklearn \
    python3-setuptools \
    python3-tk \
    python3-tqdm \
    python3-venv \
    python3-yaml \
    software-properties-common \
    tar \
    tree \
    unzip \
    vim \
    wget \
    x11-apps \
    x11-utils \
    xdg-utils \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pybullet

RUN cat <<'EOF' > /usr/local/bin/check_env
#!/bin/bash
set -e

echo "========== OS =========="
cat /etc/os-release || true

echo ""
echo "========== Compiler =========="
gcc --version | head -n 1
g++ --version | head -n 1
cmake --version | head -n 1

echo ""
echo "========== OpenCV =========="
pkg-config --modversion opencv4 || true
python3 -c "import cv2; print('python cv2:', cv2.__version__)" || true

echo ""
echo "========== PCL =========="
pkg-config --modversion pcl_common || true

echo ""
echo "========== Eigen =========="
dpkg -s libeigen3-dev | grep Version || true

echo ""
echo "========== PyBullet =========="
python3 -c "import pybullet; print('pybullet available')" || true

echo ""
echo "Environment check finished."
EOF

RUN chmod +x /usr/local/bin/check_env

RUN echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias check_env='/usr/local/bin/check_env'" >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]
CMD ["-l"]
