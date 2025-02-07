#%%
import platform
import subprocess
import uuid
def get_cpu_info():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        command = "/usr/sbin/sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command, shell=True).strip().decode()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo | grep 'model name' | uniq"
        return subprocess.check_output(command, shell=True).strip().decode()
    else:
        return "Unknown CPU"

def get_motherboard_serial():
    if platform.system() == "Windows":
        command = "wmic baseboard get serialnumber"
    elif platform.system() == "Linux":
        command = "dmidecode -s baseboard-serial-number"
    elif platform.system() == "Darwin":
        command = "ioreg -l | grep IOPlatformSerialNumber"
    else:
        return "Unknown Motherboard"
    return subprocess.check_output(command, shell=True).strip().decode()

def get_disk_serial():
    if platform.system() == "Windows":
        command = "wmic diskdrive get serialnumber"
    elif platform.system() == "Linux":
        command = "lsblk -o SERIAL | sed -n 2p"
    elif platform.system() == "Darwin":
        command = "system_profiler SPStorageDataType | grep 'Serial Number' | awk '{print $3}'"
    else:
        return "Unknown Disk"
    return subprocess.check_output(command, shell=True).strip().decode()



def get_mac_address():
    mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0,2*6,2)][::-1])
    return mac

def get_os_info():
    return platform.platform()

def get_computer_fingerprint():
    fingerprint_info = {
        "cpu_info": get_cpu_info(),
        "motherboard_serial": get_motherboard_serial(),
        "disk_serial": get_disk_serial(),
        "mac_address": get_mac_address(),
        "os_info": get_os_info(),
    }
    return fingerprint_info

print(get_computer_fingerprint())