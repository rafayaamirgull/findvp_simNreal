ARG DOCKER_REPO
ARG ROS_DISTRO
ARG IMAGE_SUFFIX
FROM $DOCKER_REPO:$ROS_DISTRO$IMAGE_SUFFIX
ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod --append --groups video $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# Install additonal packages - add any that you need
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python3-venv python-is-python3 ssh neovim git iproute2 iputils-ping python3-tqdm python3-requests \
    ros-$ROS_DISTRO-libg2o ros-$ROS_DISTRO-dwb-critics ros-$ROS_DISTRO-nav2-core ros-$ROS_DISTRO-nav2-costmap-2d \
    ros-$ROS_DISTRO-nav2-msgs ros-$ROS_DISTRO-nav2-util ros-$ROS_DISTRO-nav2-bringup ros-$ROS_DISTRO-librealsense2 \
    ros-$ROS_DISTRO-diagnostic-updater ros-$ROS_DISTRO-launch-pytest ros-$ROS_DISTRO-moveit-kinematics ros-$ROS_DISTRO-moveit-planners-ompl \
    ros-$ROS_DISTRO-moveit-ros-move-group ros-$ROS_DISTRO-moveit-ros-visualization ros-$ROS_DISTRO-moveit-servo ros-$ROS_DISTRO-moveit-simple-controller-manager \
    ros-$ROS_DISTRO-ur-description ros-$ROS_DISTRO-ur-robot-driver ros-$ROS_DISTRO-warehouse-ros-sqlite ros-$ROS_DISTRO-navigation2 ros-$ROS_DISTRO-nav2-common \
    ros-$ROS_DISTRO-slam-toolbox ros-$ROS_DISTRO-phidgets-spatial
ENV SHELL /bin/bash

# Source ROS environment automatically
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc
RUN echo "source /home/ws/install/setup.bash" >> /home/$USERNAME/.bashrc

# Set the default user
USER $USERNAME
CMD ["/bin/bash"]
