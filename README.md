# HRI: Homework 0

Dylan Losey, Virginia Tech.

In this homework assignment we will simulate a robot arm.

## Install and Run

### Ubuntu

```bash

# Download
git clone https://github.com/vt-hri/HW0.git
cd HW0

# Create and source virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pybullet

# Run the script
python main.py
```

### Windows

One way to run this is with WSL, [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).

1. Install WSL. Windows provides instructions [here](https://learn.microsoft.com/en-us/windows/wsl/setup/environment). I also recommend this [tutorial](https://www.youtube.com/watch?v=-Wg2r1lWrTc). Once installed, make sure to update and upgrade packages using:
```bash

sudo apt update && sudo apt upgrade

```

2. Open an Ubuntu terminal. Right click, and paste the code shown above in the [Ubuntu](#ubuntu) section. After the packages are installed, you should see the [Expected Output](#expected-output).

3. Install a text editing software. I use [Sublime](https://www.sublimetext.com/download), but there are many good options. You will use this text editing software to write your code.

## Expected Output

<img src="env.gif" width="750">