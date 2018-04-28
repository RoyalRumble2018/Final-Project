from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

# Following are defaults which can be overridden later on
default_args = {
    'owner': 'jerin.rajan',
    'depends_on_past': False,
    'start_date': datetime(2018, 4, 19),
    'email': ['jerinrajan23@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('ec2', default_args=default_args,schedule_interval='*/150 0 * * *')

# t1, t2, t3 and t4 are examples of tasks created using operators

t2 = BashOperator(
    task_id='part1',
    bash_command='python /home/ec2-user/airflow/dags/DownloadDataset.py',
    dag=dag)



t3 = BashOperator(
    task_id='part2',
    bash_command='python /home/ec2-user/airflow/dags/DataCleaning.py',
    dag=dag)

t4 = BashOperator(
    task_id='part3',
    bash_command='python /home/ec2-user/airflow/dags/Models.py',
    dag=dag)


t5 = BashOperator(
    task_id='part4',
    bash_command='python /home/ec2-user/airflow/dags/UploadtoS3.py',
    dag=dag)

t6 = BashOperator(
    task_id='part5',
    bash_command='docker cp /home/ec2-user/airflow/dags/Models/. 3f556d329981:/usr/local/src/assg/Models/',
    dag=dag)



t1 = BashOperator(
    task_id='Connected',
    bash_command='echo "Final Project"',
    dag=dag)



t2.set_upstream(t1)
t3.set_upstream(t2)
t4.set_upstream(t3)
t5.set_upstream(t4)
t6.set_upstream(t5)
