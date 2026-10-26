from types import SimpleNamespace

from adoption_forecaster.AI_adjustments import gcp_secrets


class _FakeClient:
    def access_secret_version(self, request):
        assert request["name"] == "projects/p/secrets/s/versions/latest"
        return SimpleNamespace(payload=SimpleNamespace(data=b" secret-from-sm \n"))


def _forbid_secret_manager(monkeypatch):
    def boom():
        raise AssertionError("Secret Manager must not be called")

    monkeypatch.setattr(gcp_secrets.secretmanager, "SecretManagerServiceClient", boom)


def test_env_var_is_used_without_calling_secret_manager(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", " env-key \n")
    _forbid_secret_manager(monkeypatch)
    assert gcp_secrets.get_openai_api_key("p", "s") == "env-key"


def test_blank_env_var_falls_back_to_secret_manager(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setattr(gcp_secrets.secretmanager, "SecretManagerServiceClient", _FakeClient)
    assert gcp_secrets.get_openai_api_key("p", "s") == "secret-from-sm"


def test_unset_env_var_reads_secret_manager(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(gcp_secrets.secretmanager, "SecretManagerServiceClient", _FakeClient)
    assert gcp_secrets.get_openai_api_key("p", "s") == "secret-from-sm"
