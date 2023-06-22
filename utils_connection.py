from paramiko import SSHClient
from scp import SCPClient
from pathlib import Path


def get_savedir_local_gum21(dataset_str_id, out_dir):
    return f"{out_dir}/gum21/{dataset_str_id}"


def get_results_gum21_RR(dataset_str_id, out_dir="RR_results"):
    # on cluster
    out_dir_cluster = f"~/py-pmi/{out_dir}"
    results_dir = out_dir_cluster + "/" + dataset_str_id

    # local
    savedir_local = get_savedir_local_gum21(dataset_str_id, out_dir)

    def check_path_exists(ssh_client, path):
        cmd = "ls " + path
        stdin, stdout, stderr = ssh_client.exec_command(cmd)
        stderr_read = stderr.read()
        is_path_exists = stderr_read.decode("UTF-8") == ""

        return is_path_exists

    def list_error_paths(ssh_client, path):
        cmd = f"ls {path}/mean_abs_errors*.npy"
        stdin, stdout, stderr = ssh_client.exec_command(cmd)
        stderr_read = stderr.read().decode("UTF-8")
        stdout_read = stdout.read().decode("UTF-8")

        if stderr_read != "":
            print("no mean abs errors files")
            print(stderr_read)
            return []
        else:
            listed_paths = stdout_read.strip().split("\n")
            return listed_paths

    def list_metadata_paths(ssh_client, path):
        cmd = f"ls {path}/metadata*.npz"
        stdin, stdout, stderr = ssh_client.exec_command(cmd)
        stderr_read = stderr.read().decode("UTF-8")
        stdout_read = stdout.read().decode("UTF-8")

        if stderr_read != "":
            print("no metadata files")
            print(stderr_read)
            return []
        else:
            listed_paths = stdout_read.strip().split("\n")
            return listed_paths

    with SSHClient() as ssh1:
        ssh1.load_system_host_keys()
        ssh1.connect(hostname="epgmod.phys.tue.nl", username="anne")

        transport = ssh1.get_transport()

        dest_addr = ("gum21", 22)
        local_addr = ("127.0.0.1", 22)
        channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)

        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect(hostname="gum21", username="anne", sock=channel)

            with SCPClient(ssh.get_transport()) as scp:
                savedir = Path(savedir_local)

                # get losses*.json
                error_paths = list_error_paths(ssh, results_dir)
                for path in error_paths:
                    if check_path_exists(ssh, path):
                        savedir.mkdir(parents=True, exist_ok=True)
                        scp.get(path, savedir)

                # get config*.json
                metadata_paths = list_metadata_paths(ssh, results_dir)
                for path in metadata_paths:
                    if check_path_exists(ssh, path):
                        savedir.mkdir(parents=True, exist_ok=True)
                        scp.get(path, savedir)
