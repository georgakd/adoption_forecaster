import copy
import os
import re
from pathlib import Path

import pytest
from google.api_core import exceptions as gapi_exceptions
from google.auth.credentials import AnonymousCredentials
from google.cloud import pubsub_v1, storage

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.main import run_single_module
from quotas_forecasting.services.modelling_files_pb2 import FileType, ModellingOutputFile
from quotas_forecasting.synthetic.writers import write_age_gender_inputs, write_education_inputs

PRED_YEAR = 2026
LAST_YEAR = 2024
DATA_SOURCE = "Eurostat"

pytestmark = [
    pytest.mark.emulators,
    pytest.mark.skipif(
        not (os.environ.get("STORAGE_EMULATOR_HOST") and os.environ.get("PUBSUB_EMULATOR_HOST")),
        reason="needs STORAGE_EMULATOR_HOST and PUBSUB_EMULATOR_HOST (see docker-compose.emulators.yml)",
    ),
]


@pytest.fixture
def emulator_config():
    config = copy.deepcopy(load_config())
    config["gcs"]["data_source"] = DATA_SOURCE
    config["pubsub"]["enable_notifications"] = True
    config["defaults"]["bucket_name"] = "ml-quotas-forecasting"
    config["defaults"]["bucket_name_external"] = "universe-bucket-staging"
    return config


def _gcs_style(name):
    """Local writers use `_` in the time part; GCS inputs carry colons."""
    return re.sub(r"T(\d{2})_(\d{2})_(\d{2})", r"T\1:\2:\3", name)


@pytest.fixture
def emulator_env(emulator_config, world, tmp_path):
    cfg = emulator_config
    client = storage.Client(project=cfg["defaults"]["project_id"], credentials=AnonymousCredentials())
    buckets = {}
    for name in (cfg["defaults"]["bucket_name"], cfg["defaults"]["bucket_name_external"]):
        try:
            client.get_bucket(name).delete(force=True)
        except gapi_exceptions.NotFound:
            pass
        buckets[name] = client.create_bucket(name)

    project, topic_id = cfg["pubsub"]["sub_project_id"], cfg["pubsub"]["topic_name"]
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    topic_path = publisher.topic_path(project, topic_id)
    sub_path = subscriber.subscription_path(project, f"{topic_id}-test-sub")
    for fn, path in ((subscriber.delete_subscription, sub_path), (publisher.delete_topic, topic_path)):
        try:
            fn(request={"subscription" if "subscriptions" in path else "topic": path})
        except gapi_exceptions.NotFound:
            pass
    publisher.create_topic(request={"name": topic_path})
    subscriber.create_subscription(request={"name": sub_path, "topic": topic_path})

    for namespace, writer in (("Education", write_education_inputs), ("Age_Gender", write_age_gender_inputs)):
        for path in writer(world, tmp_path / namespace):
            buckets[cfg["defaults"]["bucket_name"]].blob(f"{DATA_SOURCE}/{namespace}/input/{_gcs_style(Path(path).name)}").upload_from_filename(str(path))

    yield {"config": cfg, "client": client, "subscriber": subscriber, "sub_path": sub_path}
    subscriber.close()


def test_prod_pipeline_uploads_and_notifies(emulator_env):
    cfg, client = emulator_env["config"], emulator_env["client"]
    for module in ("education", "age_gender"):
        run_single_module(module, "prod", cfg["defaults"]["bucket_name"], cfg["defaults"]["project_id"], PRED_YEAR, LAST_YEAR, cfg)

    internal = [b.name for b in client.list_blobs(cfg["defaults"]["bucket_name"])]
    for namespace in ("Education", "Age_Gender"):
        assert any(n.startswith(f"{DATA_SOURCE}/{namespace}/output/{PRED_YEAR}_") for n in internal), (namespace, internal)

    external = [b.name for b in client.list_blobs(cfg["defaults"]["bucket_name_external"])]
    prefix = f"{cfg['defaults']['dir_external']}/{PRED_YEAR}_"
    assert external and all(n.startswith(prefix) and n.endswith("_with_past_data_both.csv") for n in external), external
    assert len(external) == 3

    response = emulator_env["subscriber"].pull(request={"subscription": emulator_env["sub_path"], "max_messages": 10}, timeout=30)
    messages = []
    for received in response.received_messages:
        msg = ModellingOutputFile()
        msg.ParseFromString(received.message.data)
        messages.append(msg)
    assert len(messages) == 3
    expected = {FileType.Value(t) for t in ("AGE_AND_GENDER", "EDUCATION_16_TO_64", "EDUCATION_16_TO_74")}
    assert {m.file_type for m in messages} == expected
    for m in messages:
        assert m.sender == "quotas_forecasting"
        assert m.projection_year == PRED_YEAR
