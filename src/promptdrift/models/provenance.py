"""Per-case evidence. Hashes are identifiers, not anonymization guarantees."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_hash: str
    configuration_hash: str
    contract_hash: str
    provider_hash: str
    engine_version: str
    # Optional provider-reported metadata is only a signal, never proof of drift.
    response_model_hash: str | None = None
    system_fingerprint_hash: str | None = None
    # One hash per declared request-envelope component; absent for cases without an envelope.
    envelope_hashes: dict[str, str] | None = None

    @property
    def composite_fingerprint(self) -> str:
        """Deterministic composite digest over input, configuration, contract, provider, and envelope hashes."""
        from promptdrift.engine.provenance import digest

        payload = {
            "input_hash": self.input_hash,
            "configuration_hash": self.configuration_hash,
            "contract_hash": self.contract_hash,
            "provider_hash": self.provider_hash,
            "envelope_hashes": self.envelope_hashes,
        }
        return digest(payload)

    @property
    def composite_hash(self) -> str:
        """Alias for composite_fingerprint."""
        return self.composite_fingerprint

    @model_serializer(mode="wrap")
    def omit_absent_envelope(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Keep baselines for cases without an envelope byte-identical to earlier v3 files.
        data = handler(self)
        if data.get("envelope_hashes") is None:
            data.pop("envelope_hashes", None)
        return data
