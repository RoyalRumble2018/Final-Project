import boto3
import paramiko
def worker_handler():

    s3_client = boto3.client('s3')
    #Download private key file from secure S3 bucket
    s3_client.download_file('adscsv1','airflow.pem', '/tmp/keyname.pem')

    k = paramiko.RSAKey.from_private_key_file("/tmp/keyname.pem")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    host='52.27.35.131'
    print "Connecting to " + host
    c.connect( hostname = host, username = "ec2-user", pkey = k )
    print "Connected to " + host

    commands = [
        "echo 'Jeriiiiiiiiiiiiiiiiiiiiiiisssss'"

        ]
    for command in commands:
        print "Executing {}".format(command)
        stdin , stdout, stderr = c.exec_command(command)
        print stdout.read()
        print stderr.read()

    return
    {
        'message' : "Script execution completed. See Cloudwatch logs for complete output"
    }

worker_handler()
