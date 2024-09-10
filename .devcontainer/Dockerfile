# Use the base image
FROM nvcr.io/nvidia/deepstream:7.0-gc-triton-devel

# Set the working directory
WORKDIR /opt/nvidia/deepstream/deepstream

# Run the installation script
RUN ./user_deepstream_python_apps_install.sh --build-bindings

# Install OpenCV - used to build Deepstream output parser
RUN apt update && apt install -y libopencv-dev python3-opencv
