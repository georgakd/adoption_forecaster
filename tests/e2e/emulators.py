"""Helpers for the emulator tests (the fixtures live in conftest.py)."""

import time

from google.api_core import exceptions as gapi_exceptions

from adoption_forecaster.services.modelling_files_pb2 import FileType, ModellingOutputFile

DATA_SOURCE = "Eurostat"
PULL_DEADLINE_SECONDS = 30
PRED_YEAR = 2026


def pull_messages(subscriber, sub_path, expected, deadline_seconds=PULL_DEADLINE_SECONDS):
    """Pull (and ack) until ``expected`` messages arrived or the deadline passed, then one short extra pull so surplus messages
    are still counted. Returns the raw message payloads in arrival order."""
    payloads = []
    deadline = time.monotonic() + deadline_seconds
    extra_pull_done = False
    while not extra_pull_done:
        if len(payloads) >= expected or time.monotonic() >= deadline:
            extra_pull_done = True
            timeout = 2
        else:
            timeout = max(1, deadline - time.monotonic())
        try:
            response = subscriber.pull(request={"subscription": sub_path, "max_messages": 10}, timeout=timeout)
        except gapi_exceptions.DeadlineExceeded:
            continue
        if response.received_messages:
            subscriber.acknowledge(request={"subscription": sub_path, "ack_ids": [r.ack_id for r in response.received_messages]})
        payloads += [r.message.data for r in response.received_messages]
    return payloads


def assert_uploaded_and_notified(emulator_env):
    """Internal outputs for both namespaces, exactly the 3 external files, and exactly 3 notifications (one per external file)."""
    cfg, client = emulator_env["config"], emulator_env["client"]
    internal = [b.name for b in client.list_blobs(cfg["defaults"]["bucket_name"])]
    for namespace in ("Education", "Age_Gender"):
        assert any(n.startswith(f"{DATA_SOURCE}/{namespace}/output/{PRED_YEAR}_") for n in internal), (namespace, internal)

    external = [b.name for b in client.list_blobs(cfg["defaults"]["bucket_name_external"])]
    prefix = f"{cfg['defaults']['dir_external']}/{PRED_YEAR}_"
    assert external and all(n.startswith(prefix) and n.endswith("_with_past_data_both.csv") for n in external), external
    assert len(external) == 3

    messages = []
    for data in pull_messages(emulator_env["subscriber"], emulator_env["sub_path"], expected=3):
        msg = ModellingOutputFile()
        msg.ParseFromString(data)
        messages.append(msg)
    assert len(messages) == 3
    expected = {FileType.Value(t) for t in ("AGE_AND_GENDER", "EDUCATION_16_TO_64", "EDUCATION_16_TO_74")}
    assert {m.file_type for m in messages} == expected
    for m in messages:
        assert m.sender == "adoption_forecaster"
        assert m.projection_year == PRED_YEAR
