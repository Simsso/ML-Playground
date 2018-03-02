#!/usr/bin/env python

# Source: Martin Thoma at https://stackoverflow.com/questions/42326466
# Runs on Linux (not Mac)

# core modules
import subprocess


def get_log_string():
    """
    Get important environment information that might influence experiments.

    Returns
    -------
    log_string : str
    """
    log_string = []
    with open('/proc/cpuinfo') as f:
        cpuinfo = f.readlines()
    for line in cpuinfo:
        if "model name" in line:
            log_string.append("CPU: {}".format(line.strip()))
            break

    with open('/proc/driver/nvidia/version') as f:
        version = f.read().strip()
    log_string.append("GPU driver: {}".format(version))
    log_string.append("VGA: {}".format(find_vga()))
    return "\n".join(log_string)


def find_vga():
    vga = subprocess.check_output("lspci | grep -i 'vga\|3d\|2d'",
                                  shell=True,
                                  executable='/bin/bash')
    return vga


print(get_log_string())