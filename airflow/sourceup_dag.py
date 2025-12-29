from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG("sourceup",start_date=datetime(2024,1,1),schedule="@daily") as dag:
    run=BashOperator(
        task_id="pipeline",
        bash_command="python pipeline/run_all.py"
    )
