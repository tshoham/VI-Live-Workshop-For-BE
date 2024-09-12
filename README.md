# VI-Live-Workshop-For-BE

Welcome to VI Live workshop!
You will get to know Nvidia's deepstream in this hand on workshop. 
The labs will demonstrate the most basic features and capabilities.

Hopefully this will give you a better understanding of the amazing features and capabilities of Nvidia's Deepstream, and how amazing VI Live will be.

## Prerequisites

Connect to AzVpn

## Workshop Content

[Workshop Setup](#Setup)
- [Lab 1 - Creating 4 different pipelines](./src/Lab1)
- [Lab 2 - Creating a pipeline with multiple inputs](./src/Lab2)

## Setup

### Steps

We will use [ts-gpu-deepstream](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/24237b72-8546-4da5-b204-8c3cb76dd930/resourceGroups/yl-try-rg/providers/Microsoft.Compute/virtualMachines/ts-gpu-deepstream/overview)

#### Setting up key and verifying connection to the VM

1. In the Go Help -> Reset password and
1. Choose "Add SSH public key"
1. **Username**: azureuser
1. **SSH public key source**: Generate new key pair
1. **SSH Key Type**: Ed25519
1. **Key Pair name**: ts-gpu-deepstream-key
1. Click **Update**
1. A window will popup allowing you to download the key.
   
    Download and save the key under C:\Users\\\<username>\\\.ssh\\\ts-gpu-deepstream-key

#### Connecting to the VM from VSCode

1. Connect to AzVpn
1. In VSCode, on the bottom left, click on the blue double arrow >< sign
![alt text](image.png)
1. Click "Connect to host"
1. Click on "Configure SSH Hosts" (choose the file that starts with "C:Users...")
1. Update config file  and save with path to key:
      ```
      Host 40.124.109.198
         HostName 40.124.109.198
         User azureuser
         IdentityFile ~/.ssh/ts-gpu-deepstream-key.pem
      ```
1. Paste your user@ip (for example azureuser@40.124.109.198)
1. Choose linux os
1. Choose "Open folder" choose path to "VI-Live-Workshop-For-BE"

#### Connecting to Dev Container within the VM

ctrl+shift+p and choose "Dev Containers: Open Folder in container"
 
> [!TIP]
> If you used a different/new user when creating the key run ```sudo usermod -a -G docker azureuser```. By adding a user to the docker group, you allow this user to run Docker commands without needing to use sudo each time.