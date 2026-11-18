"""Request-envelope provenance: deterministic, credential-safe, component-attributed."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from promptdrift.engine.baseline import load_baseline, write_baseline
from promptdrift.engine.monitor import monitor_suite
from promptdrift.engine.provenance import (
    composite_provenance_fingerprint,
    envelope_hashes,
    fingerprint,
    strip_credentials,
)
from promptdrift.engine.runner import run_suite
from promptdrift.models import Config
from promptdrift.models.result import ModelResponse
from promptdrift.models.test import RequestEnvelope
from promptdrift.providers.mock import MockProvider


def envelope(**overrides):
    data = {
        "system": "You are a support agent.",
        "tools": [
            {
                "name": "lookup_order",
                "description": "Find an order by id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "max_tokens": {"type": "integer"},
                    },
                    "required": ["order_id"],
                },
            }
        ],
        "retrieval_context": {"policy": {"refund_days": 30, "region": "EU"}, "faq": ["a", "b"]},
    }
    data.update(overrides)
    return RequestEnvelope.model_validate(data)


def configured(tmp_path, request_envelope=None):
    test = {
        "id": "policy",
        "prompt": "policy.txt",
        "variables": {"days": 30},
        "assertions": [{"type": "contains", "value": "30 days"}],
    }
    if request_envelope is not None:
        test["request_envelope"] = request_envelope
    config = Config.model_validate({"provider": {"type": "mock", "model": "echo"}, "tests": [test]})
    (tmp_path / "policy.txt").write_text("Refunds: {{ days }} days", encoding="utf-8")
    path = tmp_path / "promptdrift.yaml"
    path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
    return config, path


def test_hashes_ignore_key_order_including_nested():
    first = envelope()
    second = RequestEnvelope.model_validate(
        {
            "retrieval_context": {"faq": ["a", "b"], "policy": {"region": "EU", "refund_days": 30}},
            "tools": [
                {
                    "parameters": {
                        "required": ["order_id"],
                        "properties": {
                            "max_tokens": {"type": "integer"},
                            "order_id": {"type": "string"},
                        },
                        "type": "object",
                    },
                    "description": "Find an order by id.",
                    "name": "lookup_order",
                }
            ],
            "system": "You are a support agent.",
        }
    )
    assert envelope_hashes(first) == envelope_hashes(second)


@pytest.mark.parametrize(
    ("component", "value"),
    [
        ("system", "You are a billing agent."),
        ("tools", [{"name": "lookup_order", "description": "Find an order."}]),
        ("retrieval_context", {"policy": {"refund_days": 14, "region": "EU"}, "faq": ["a", "b"]}),
        ("hyperparameters", {"temperature": 0.2}),
    ],
)
def test_each_component_changes_only_its_own_hash(component, value):
    before = envelope_hashes(envelope())
    after = envelope_hashes(envelope(**{component: value}))
    assert {name for name in before if before[name] != after[name]} == {component}


def test_credentials_do_not_affect_hashes():
    def with_secrets(key, bearer):
        return envelope(
            tools=[
                {
                    "name": "lookup_order",
                    "parameters": {"api_key": key, "headers": {"Authorization": bearer}},
                }
            ],
            retrieval_context={"X-API-Key": key, "notes": f"use {key} here", "source": "kb"},
        )

    first = with_secrets("sk-" + "a" * 32, "Bearer " + "b" * 32)
    second = with_secrets("sk-" + "c" * 32, "Bearer " + "d" * 32)
    assert envelope_hashes(first) == envelope_hashes(second)
    stripped = json.dumps(strip_credentials(first.model_dump(mode="json")))
    assert "a" * 32 not in stripped and "b" * 32 not in stripped
    assert "api_key" not in stripped and "Authorization" not in stripped


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "privateKey",
        "refresh_token",
        "session-token",
        "cookie",
        "Set-Cookie",
        "auth_token",
        "authToken",
        "api-secret",
        "apiSecret",
        "access_key",
        "accessKey",
        "client_key",
        "clientKey",
    ],
)
def test_sensitive_keys_are_removed_case_and_separator_insensitively(key):
    assert strip_credentials({key: "value", "kept": 1}) == {"kept": 1}


def test_schema_property_named_like_a_credential_still_counts():
    def password_tool(kind):
        return envelope(
            tools=[{"name": "login", "parameters": {"properties": {"password": {"type": kind}}}}]
        )

    assert envelope_hashes(password_tool("string")) != envelope_hashes(password_tool("integer"))
    assert strip_credentials({"max_tokens": 5}) == {"max_tokens": 5}


@pytest.mark.parametrize("text", ["risk-assessment", "task-scheduler.v2"])
def test_ordinary_text_is_never_redacted(text):
    assert strip_credentials(text) == text


def test_editing_a_risk_description_changes_the_tools_hash():
    def described(text):
        return envelope(tools=[{"name": "triage", "description": text}])

    before = envelope_hashes(described("Run the risk-assessment workflow."))
    after = envelope_hashes(described("Run the risk-evaluation workflow."))
    assert before["tools"] != after["tools"]


TOOLS = [
    {"name": "lookup_order", "description": "Find an order.", "parameters": {"type": "object"}},
    {"name": "refund_order", "description": "Refund an order.", "parameters": {"type": "object"}},
]


def test_tool_declaration_order_does_not_change_tools_hash():
    declared = envelope(tools=TOOLS)
    reordered = envelope(tools=list(reversed(TOOLS)))
    assert envelope_hashes(declared) == envelope_hashes(reordered)
    # Sorting happens only while hashing; the model keeps the declared order.
    assert [tool.name for tool in reordered.tools] == ["refund_order", "lookup_order"]


@pytest.mark.parametrize(
    "change",
    [
        {"name": "cancel_order"},
        {"description": "Find an order by id."},
        {"parameters": {"type": "object", "required": ["order_id"]}},
    ],
)
def test_changing_a_tool_changes_tools_hash(change):
    changed = [{**TOOLS[0], **change}, TOOLS[1]]
    assert (
        envelope_hashes(envelope(tools=TOOLS))["tools"]
        != (envelope_hashes(envelope(tools=changed))["tools"])
    )


def test_lists_inside_tool_parameters_keep_declared_order():
    def tool(required):
        return envelope(tools=[{"name": "t", "parameters": {"required": required}}])

    assert envelope_hashes(tool(["a", "b"]))["tools"] != envelope_hashes(tool(["b", "a"]))["tools"]


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        envelope(tools=[{"name": "a"}, {"name": "a"}])


def test_case_without_envelope_keeps_previous_provenance(tmp_path):
    config, path = configured(tmp_path)
    provenance = fingerprint(config, config.tests[0], "Refunds: 30 days")
    assert provenance.envelope_hashes is None
    assert set(json.loads(provenance.model_dump_json())) == {
        "input_hash",
        "configuration_hash",
        "contract_hash",
        "provider_hash",
        "engine_version",
        "response_model_hash",
        "system_fingerprint_hash",
    }
    with_envelope = config.tests[0].model_copy(
        update={"request_envelope": RequestEnvelope(system="x")}
    )
    other = fingerprint(config, with_envelope, "Refunds: 30 days")
    assert other.model_dump(exclude={"envelope_hashes"}) == provenance.model_dump(
        exclude={"envelope_hashes"}
    )


def test_old_baseline_without_envelope_loads_and_reports_no_envelope_drift(tmp_path):
    config, path = configured(tmp_path)
    baseline_path = tmp_path / config.baseline.path
    write_baseline(baseline_path, run_suite(config, path))
    assert "envelope_hashes" not in baseline_path.read_text(encoding="utf-8")
    assert load_baseline(baseline_path).tests["policy"].provenance.envelope_hashes is None
    case = monitor_suite(config, path, samples=2).tests[0]
    assert case.diagnosis == "stable"
    assert not any("request_envelope" in item for item in case.evidence)


def test_envelope_hashes_round_trip_through_baseline(tmp_path):
    config, path = configured(tmp_path, envelope().model_dump(mode="json"))
    baseline_path = tmp_path / config.baseline.path
    write_baseline(baseline_path, run_suite(config, path))
    stored = load_baseline(baseline_path).tests["policy"].provenance.envelope_hashes
    assert stored == envelope_hashes(envelope())
    assert monitor_suite(config, path, samples=2).tests[0].diagnosis == "stable"


def test_monitor_evidence_names_the_drifted_component(tmp_path):
    config, path = configured(tmp_path, envelope().model_dump(mode="json"))
    write_baseline(tmp_path / config.baseline.path, run_suite(config, path))
    config.tests[0].request_envelope = envelope(tools=[{"name": "refund_order"}])
    case = monitor_suite(config, path, samples=2).tests[0]
    assert case.diagnosis == "prompt_changed"
    envelope_evidence = [item for item in case.evidence if "request_envelope" in item]
    assert envelope_evidence == ["request_envelope.tools differs from the accepted baseline."]


def test_adding_an_envelope_after_baselining_is_reported(tmp_path):
    config, path = configured(tmp_path)
    write_baseline(tmp_path / config.baseline.path, run_suite(config, path))
    config.tests[0].request_envelope = envelope()
    case = monitor_suite(config, path, samples=2).tests[0]
    assert case.diagnosis == "prompt_changed"
    assert "request_envelope was added since the accepted baseline." in case.evidence


def test_default_envelope_construction():
    env = RequestEnvelope()
    assert env.version == 1
    assert env.system is None
    assert env.tools == []
    assert env.retrieval_context == {}
    assert env.hyperparameters == {}
    assert env.uncaptured_dimensions == []
    assert env.temperature is None
    assert env.top_p is None
    assert env.response_format is None

    # Version must be >= 1
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        RequestEnvelope(version=0)

    # Extra fields forbidden
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RequestEnvelope(extra_field="invalid")


def test_shorthand_hyperparameters_sync():
    env = RequestEnvelope(temperature=0.7, top_p=0.9, response_format="json_object")
    assert env.hyperparameters == {
        "temperature": 0.7,
        "top_p": 0.9,
        "response_format": "json_object",
    }

    # Explicit hyperparameters are not overwritten by shorthand
    env2 = RequestEnvelope(temperature=0.7, hyperparameters={"temperature": 0.5})
    assert env2.hyperparameters["temperature"] == 0.5


def test_uncaptured_dimensions_handling():
    env_list = RequestEnvelope(uncaptured_dimensions=["user_locale", "session_id"])
    assert env_list.uncaptured_dimensions == ["user_locale", "session_id"]
    env_dict = RequestEnvelope(uncaptured_dimensions={"user_locale": "en_US"})
    assert env_dict.uncaptured_dimensions == {"user_locale": "en_US"}


def test_token_redaction_regex_case_and_length():
    # >= 16 chars sk- and Bearer tokens are redacted
    assert strip_credentials("sk-1234567890abcdef") == "[REDACTED]"
    assert strip_credentials("bearer 1234567890abcdef") == "[REDACTED]"
    assert strip_credentials("BEARER 1234567890abcdef") == "[REDACTED]"
    # Shorter strings are not matched
    assert strip_credentials("sk-short") == "sk-short"
    assert strip_credentials("Bearer short") == "Bearer short"


def test_composite_provenance_fingerprint_determinism():
    env1 = envelope()
    env2 = RequestEnvelope.model_validate(
        {
            "tools": list(reversed(env1.tools)),
            "system": env1.system,
            "retrieval_context": env1.retrieval_context,
            "hyperparameters": env1.hyperparameters,
            "uncaptured_dimensions": ["beta", "alpha"],
        }
    )
    env1_with_dims = env1.model_copy(update={"uncaptured_dimensions": ["alpha", "beta"]})

    fp1 = composite_provenance_fingerprint("Prompt text", {"var": 1}, env1_with_dims)
    fp2 = composite_provenance_fingerprint("Prompt text", {"var": 1}, env2)
    # Permuting tool order and uncaptured_dimensions produces identical composite fingerprint
    assert fp1 == fp2
    assert len(fp1) == 64

    # Changing prompt changes composite fingerprint
    assert composite_provenance_fingerprint("Other text", {"var": 1}, env1) != fp1
    # Changing variables changes composite fingerprint
    assert composite_provenance_fingerprint("Prompt text", {"var": 2}, env1) != fp1
    # Redacted credentials in variables do not change composite fingerprint
    fp_creds1 = composite_provenance_fingerprint("Prompt text", {"api_key": "secret1", "var": 1}, env1)
    fp_creds2 = composite_provenance_fingerprint("Prompt text", {"api_key": "secret2", "var": 1}, env1)
    assert fp_creds1 == fp_creds2
    # Without envelope works cleanly
    fp_no_env = composite_provenance_fingerprint("Prompt text", {"var": 1})
    assert len(fp_no_env) == 64
    assert fp_no_env != fp1


def test_provenance_composite_fingerprint_and_hash_properties(tmp_path):
    config, path = configured(tmp_path, envelope().model_dump(mode="json"))
    prov = fingerprint(config, config.tests[0], "Refunds: 30 days")
    assert prov.composite_fingerprint == prov.composite_hash
    assert len(prov.composite_fingerprint) == 64

    # Different prompt changes input_hash and composite_fingerprint
    prov2 = fingerprint(config, config.tests[0], "Different prompt")
    assert prov.composite_fingerprint != prov2.composite_fingerprint

    # Property is excluded from serialized json (preserves byte-identical baseline)
    serialized = json.loads(prov.model_dump_json())
    assert "composite_fingerprint" not in serialized
    assert "composite_hash" not in serialized


def test_legacy_baseline_without_hyperparameters_compatibility(tmp_path):
    config, path = configured(tmp_path, envelope().model_dump(mode="json"))
    report = run_suite(config, path)
    # Simulate legacy baseline created without hyperparameters hash
    legacy_hashes = {k: v for k, v in report.tests[0].provenance.envelope_hashes.items() if k != "hyperparameters"}
    report.tests[0].provenance.envelope_hashes = legacy_hashes
    baseline_path = tmp_path / config.baseline.path
    write_baseline(baseline_path, report)

    # Monitoring with current code does not falsely attribute drift
    case = monitor_suite(config, path, samples=2).tests[0]
    assert case.diagnosis == "stable"
    assert not any("hyperparameters" in item for item in case.evidence)


def test_envelope_removed_after_baselining_is_reported(tmp_path):
    config, path = configured(tmp_path, envelope().model_dump(mode="json"))
    write_baseline(tmp_path / config.baseline.path, run_suite(config, path))
    config.tests[0].request_envelope = None
    case = monitor_suite(config, path, samples=2).tests[0]
    assert case.diagnosis == "prompt_changed"
    assert "request_envelope was removed since the accepted baseline." in case.evidence


def test_monitor_hyperparameters_drift_reported(tmp_path):
    config, path = configured(
        tmp_path,
        envelope(hyperparameters={"temperature": 0.2}).model_dump(mode="json"),
    )
    write_baseline(tmp_path / config.baseline.path, run_suite(config, path))
    config.tests[0].request_envelope = envelope(hyperparameters={"temperature": 0.8})
    case = monitor_suite(config, path, samples=2).tests[0]
    assert case.diagnosis == "prompt_changed"
    assert "request_envelope.hyperparameters differs from the accepted baseline." in case.evidence


def test_observed_model_drift_with_envelope(tmp_path, monkeypatch):
    config, path = configured(tmp_path, envelope().model_dump(mode="json"))
    write_baseline(tmp_path / config.baseline.path, run_suite(config, path))

    # Provenance (prompt, settings, contracts, envelope) remains identical,
    # but model repeatedly returns failing output across probes.
    def mock_failing_complete(self, prompt, **kwargs):
        return ModelResponse(
            output="No refund under any circumstances.",
            latency_ms=1.0,
            model="echo",
            provider="mock",
        )

    monkeypatch.setattr(MockProvider, "complete", mock_failing_complete)
    case = monitor_suite(config, path, samples=2).tests[0]
    assert case.diagnosis == "observed_model_drift"
