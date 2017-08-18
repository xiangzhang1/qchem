#!/usr/bin/python
import paramiko

ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.connect('nanaimo', username='xzhang1')

ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("squeue -n ss")
print ssh_stdout.readlines()
