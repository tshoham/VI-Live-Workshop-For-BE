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

## Setup

### Steps

We will use [ts-gpu-deepstream](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/24237b72-8546-4da5-b204-8c3cb76dd930/resourceGroups/yl-try-rg/providers/Microsoft.Compute/virtualMachines/ts-gpu-deepstream/overview)

#### Setting up key and verifying connection to the VM

1. In the Go Help -> Reset password and "Add SSH public key
   - Username: it will be easier to use the username azureuser in the existing vms. If creating a new oser/on a new vm, you will need to add rbac to the user (in the comments below)
   - Type: Ed25519

2. Download and save tje key under C:\Users\\\<username>\\\.ssh\\\<filename>

#### Connecting to the VM from VSCode

1. In VSCode, on the bottom left, click on the double arrow >< sign and choose "Connect to host"
2. Paste your user@ip (for example azureuser@172.179.37.123)
   - You can test your connection to the VM from bash/powershell

3. Update config file with path to key:
      ```
      Host 172.179.37.123
         HostName 172.179.37.123
         User azureuser
         IdentityFile ~/.ssh/<filename>
      ```

4. Choose linux os
5. Choose "Open folder" choose path to "VI-Live-Workshop-For-BE"

#### Connecting to Dev Container within the VM

1. Make sure the mounts has absolute path to data (update the env file)
2. ctrl+shift+p and choose "Dev Containers: Open Folder in container"
 
> [!TIP]
> If you use a different/new user when creating the key run ```sudo usermod -a -G docker azureuser```. By adding a user to the docker group, you allow this user to run Docker commands without needing to use sudo each time.