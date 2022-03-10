import os
import paramiko


username = input("Username: ")
password = input("Password: ")


ssh = paramiko.SSHClient() 
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
ssh.connect("gpucluster.st.lab.au.dk", username=username, password=password)
sftp = ssh.open_sftp()
sftp.put("C:/Users/Aksel/Desktop/test.txt", "/data")
sftp.close()
ssh.close()