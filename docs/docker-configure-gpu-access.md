# GPU Configuration Guide for Docker

## Overview

This project utilizes GPU capabilities for enhanced performance in computing tasks. This guide details the steps to configure Docker containers on Ubuntu 22.04 to access GPU devices, specifically focusing on `/dev/dri/renderD*` devices. This configuration is crucial for ensuring that Docker containers can efficiently use GPU resources.

## Prerequisites

- Ubuntu 22.04 host with GPU capabilities.
- Docker installed on the host.

## Configuration Steps

### Check Group Ownership of Render Nodes

Determine the group assigned to the render nodes on your host system:

```bash
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```

Example output:

```
group_name=render group_id=134
```

### Configure Container for Non-Root User Access

To enable GPU access for a non-root user (named `user`) in the container, follow these steps:

1. **Run the Docker Container with Group Access**:
   
   Use the group ID from your host to grant `user` access to the GPU device:

   ```bash
   docker run -it --rm --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) <image_name>
   ```

   This command adds the container's `user` to the group with the same group ID as the host's render group, allowing GPU access.

2. **Bind Mount `/dev/dri`** (Optional):

   If your application requires direct access to the device files, you can bind mount the `/dev/dri` directory:

   ```bash
   docker run -it --rm --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) -v /dev/dri:/dev/dri <image_name>
   ```

## Conclusion

By implementing this configuration, the `user` within your container, Ubuntu 22.04 will have appropriate access to the GPU resources, ensuring optimal performance for your AI and machine learning projects.
