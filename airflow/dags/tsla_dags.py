from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import os

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

SCRIPTS_DIR = "/opt/airflow/scripts"


with DAG(
    dag_id="tsla_scraper_and_snapshot_pipeline_dynamic",
    default_args=default_args,
    description="Automatically load and run all scraper scripts, then build Qdrant snapshot.",
    schedule_interval="@daily",
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["tsla", "selenium", "gcp", "qdrant", "rag", "scrapers"],
) as dag:

    start = EmptyOperator(task_id="start")
    join_scrapers = EmptyOperator(task_id="join_scrapers", trigger_rule="all_success")

    scraper_tasks = []

    # =======================================================
    # 🔥 Automatically detect all .py files in /scripts
    # =======================================================
    for fname in os.listdir(SCRIPTS_DIR):
        if not fname.endswith(".py"):
            continue  # skip non-Python files

        full_path = os.path.join(SCRIPTS_DIR, fname)

        task_id = f"run_{os.path.splitext(fname)[0].replace('-', '_')}"
        t = BashOperator(
            task_id=task_id,
            bash_command=f"python {full_path}",
        )

        start >> t >> join_scrapers
        scraper_tasks.append(t)

    # =======================================================
    # Wait for EventArc → BigQuery to sync
    # =======================================================
    wait_for_bigquery = BashOperator(
        task_id="wait_for_bigquery_sync",
        bash_command="sleep 60",
        trigger_rule="all_success",
    )

    # =======================================================
    # Build Qdrant RAG snapshot
    # =======================================================
    build_snapshot = BashOperator(
        task_id="build_qdrant_snapshot",
        bash_command=(
            "curl -X POST -H 'Content-Type: application/json' "
            "'https://tsla-agent-api-358918971535.us-central1.run.app/embed-snapshot'"
        ),
        trigger_rule="all_success",
    )

    join_scrapers >> wait_for_bigquery >> build_snapshot
