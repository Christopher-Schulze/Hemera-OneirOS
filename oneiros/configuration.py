"""Configuration loading and validation for Hemera OneirOS."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


class Profile(str, Enum):
    """Execution profile presets for the core vm."""

    STANDARD = "standard"
    CONTINUATIONS = "continuations"
    DISTRIBUTED = "distributed"


class ISA(str, Enum):
    """Guest ISA options supported by OneirOS."""

    RV32IM = "rv32im"
    RV64IM = "rv64im"
    EVM = "evm"
    WASM = "wasm"
    SVM = "svm"


class IsaMode(str, Enum):
    STANDARD = "standard"
    PER_INSTRUCTION_AIR = "per_instruction_air"


class IVCBackend(str, Enum):
    SUPERNOVA = "supernova"
    HYPERNOVA = "hypernova"
    HYPERNOVA_PROTOGALAXY = "hypernova_protogalaxy"


class RecursionCurve(str, Enum):
    PASTA = "pasta"
    BLS12_377 = "bls12_377"
    BN254 = "bn254"


class MemoryModel(str, Enum):
    SPARSE_MERKLE_TWIST = "sparse_merkle_twist"
    FULL_MERKLE = "full_merkle"


class Transcript(str, Enum):
    POSEIDON2_V2 = "poseidon2_v2"


class LookupBackend(str, Enum):
    LASSO_DECOMPOSABLE = "lasso_decomposable"
    LASSO = "lasso"
    CAULK = "caulk"


class HardwareBackend(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    FPGA = "fpga"


class LDT(str, Enum):
    DEEPFRI = "deepfri"
    STIR = "stir"
    WHIR = "whir"


class StarkField(str, Enum):
    M31 = "m31"
    BLS12_381_FR = "bls12_381_fr"


class PCS(str, Enum):
    DEEPFOLD = "deepfold"
    BINIUS = "binius"


class Folding(str, Enum):
    SUPERNOVA = "supernova"
    HYPERNOVA = "hypernova"
    PROTOGALAXY_OPTIONAL = "protogalaxy_optional"


class Wrapper(str, Enum):
    GROTH16 = "groth16"
    PLONKY3 = "plonky3"
    SUPERSONIC = "supersonic"


class Curve(str, Enum):
    BLS12_381 = "bls12_381"
    BN254 = "bn254"
    PALLAS = "pallas"
    VESTA = "vesta"


class Aggregation(str, Enum):
    SNARKPACK_CP_SNARK = "snarkpack_cp_snark"
    NOVA_FOLD = "nova_fold"


class BlobBinding(str, Enum):
    EIP4844_KZG = "eip4844_kzg"
    CELESTIA_NMT = "celestia_nmt"


@dataclass(frozen=True)
class ProfileMetadata:
    """Metadata describing a built-in profile configuration."""

    profile: "Profile"
    description: str
    defaults: Mapping[str, Any]
    target_cycles: int
    working_set_bytes: int
    concurrency: int

    def defaults_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the default configuration mapping."""

        return _deep_copy_mapping(self.defaults)

    def summary(self) -> Dict[str, Any]:
        """Return a serialisable summary of the metadata."""

        return {
            "profile": self.profile.value,
            "description": self.description,
            "target_cycles": self.target_cycles,
            "working_set_bytes": self.working_set_bytes,
            "concurrency": self.concurrency,
        }


@dataclass(frozen=True)
class OneirosSection:
    profile: Profile
    isa: ISA
    isa_mode: IsaMode
    zero_knowledge: bool
    ivc_backend: IVCBackend
    recursion_curve: RecursionCurve
    memory_model: MemoryModel
    page_size: int
    max_cycles_per_segment: int
    transcript: Transcript
    security_bits: int
    lookup_backend: LookupBackend
    hardware_backend: HardwareBackend

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OneirosSection":
        return cls(
            profile=_require_enum(Profile, data, "profile"),
            isa=_require_enum(ISA, data, "isa"),
            isa_mode=_require_enum(IsaMode, data, "isa_mode"),
            zero_knowledge=_require_bool(data, "zero_knowledge"),
            ivc_backend=_require_enum(IVCBackend, data, "ivc_backend"),
            recursion_curve=_require_enum(RecursionCurve, data, "recursion_curve"),
            memory_model=_require_enum(MemoryModel, data, "memory_model"),
            page_size=_require_positive_int(data, "page_size"),
            max_cycles_per_segment=_require_positive_int(data, "max_cycles_per_segment"),
            transcript=_require_enum(Transcript, data, "transcript"),
            security_bits=_require_positive_int(data, "security_bits"),
            lookup_backend=_require_enum(LookupBackend, data, "lookup_backend"),
            hardware_backend=_require_enum(HardwareBackend, data, "hardware_backend"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _section_to_dict(self)


@dataclass(frozen=True)
class ProvingConfig:
    ldt: LDT
    stark_field: StarkField
    pcs: PCS
    folding: Folding

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvingConfig":
        return cls(
            ldt=_require_enum(LDT, data, "ldt"),
            stark_field=_require_enum(StarkField, data, "stark_field"),
            pcs=_require_enum(PCS, data, "pcs"),
            folding=_require_enum(Folding, data, "folding"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _section_to_dict(self)


@dataclass(frozen=True)
class InjectorConfig:
    wrapper: Wrapper
    primary_curve: Curve
    fallback_curve: Curve
    aggregation: Aggregation
    blob_binding: BlobBinding

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InjectorConfig":
        primary = _require_enum(Curve, data, "primary_curve")
        fallback = _require_enum(Curve, data, "fallback_curve")
        if primary == fallback:
            raise ValueError("primary_curve and fallback_curve must differ to provide redundancy")
        return cls(
            wrapper=_require_enum(Wrapper, data, "wrapper"),
            primary_curve=primary,
            fallback_curve=fallback,
            aggregation=_require_enum(Aggregation, data, "aggregation"),
            blob_binding=_require_enum(BlobBinding, data, "blob_binding"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _section_to_dict(self)


@dataclass(frozen=True)
class AppConfig:
    oneiros: OneirosSection
    proving: ProvingConfig
    injector: InjectorConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        if "oneiros" not in data:
            raise ValueError("Configuration missing 'oneiros' section")
        if "proving" not in data:
            raise ValueError("Configuration missing 'proving' section")
        if "injector" not in data:
            raise ValueError("Configuration missing 'injector' section")
        return cls(
            oneiros=OneirosSection.from_dict(_require_mapping(data, "oneiros")),
            proving=ProvingConfig.from_dict(_require_mapping(data, "proving")),
            injector=InjectorConfig.from_dict(_require_mapping(data, "injector")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oneiros": _section_to_dict(self.oneiros),
            "proving": _section_to_dict(self.proving),
            "injector": _section_to_dict(self.injector),
        }


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Load an :class:`AppConfig` from ``path`` applying optional overrides."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    raw_data = _load_yaml(config_path)
    if not isinstance(raw_data, MutableMapping):
        raise ValueError("Configuration root must be a mapping")

    merged = dict(raw_data)
    if overrides:
        merged = _apply_overrides(merged, overrides)

    return AppConfig.from_dict(merged)


def build_profile_config(
    profile: Profile, overrides: Mapping[str, Any] | None = None
) -> AppConfig:
    """Construct an :class:`AppConfig` for a built-in profile."""

    metadata = profile_metadata(profile)
    base = metadata.defaults_dict()
    if overrides:
        base = _apply_overrides(base, overrides)
    return AppConfig.from_dict(base)


def available_profiles() -> Dict[Profile, ProfileMetadata]:
    """Return metadata for all built-in profiles."""

    return dict(_PROFILE_REGISTRY)


def profile_metadata(profile: Profile) -> ProfileMetadata:
    """Fetch metadata for ``profile``."""

    try:
        return _PROFILE_REGISTRY[profile]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown profile: {profile!r}") from exc


def _apply_overrides(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = yaml.safe_load(yaml.safe_dump(base))  # deep copy
    for key, value in overrides.items():
        parts = key.split(".")
        if not parts:
            raise ValueError("Override key must not be empty")
        cursor: Dict[str, Any] = result
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _require_enum(enum_cls: type[Enum], data: Mapping[str, Any], field: str) -> Enum:
    value = _require_value(data, field)
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for member in enum_cls:
            if member.value == normalized:
                return member
    raise ValueError(f"Invalid value '{value}' for {field}; expected one of {[m.value for m in enum_cls]}")


def _require_bool(data: Mapping[str, Any], field: str) -> bool:
    value = _require_value(data, field)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Invalid boolean for {field}: {value!r}")


def _require_positive_int(data: Mapping[str, Any], field: str) -> int:
    value = _require_value(data, field)
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid integer for {field}")
    if isinstance(value, (int, float)) and int(value) == value:
        ivalue = int(value)
    elif isinstance(value, str) and value.strip():
        try:
            ivalue = int(value.replace("_", ""), 10)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid integer for {field}: {value!r}") from exc
    else:
        raise ValueError(f"Invalid integer for {field}: {value!r}")
    if ivalue <= 0:
        raise ValueError(f"{field} must be > 0")
    return ivalue


def _require_mapping(data: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = _require_value(data, field)
    if not isinstance(value, Mapping):
        raise ValueError(f"Field {field} must be a mapping")
    return value


def _require_value(data: Mapping[str, Any], field: str) -> Any:
    if field not in data:
        raise ValueError(f"Missing required field: {field}")
    return data[field]


def _deep_copy_mapping(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a deep copy of ``data`` using YAML round-tripping."""

    return yaml.safe_load(yaml.safe_dump(data))


def _section_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dict__"):
        raw = obj.__dict__
    else:  # pragma: no cover - fall back for dataclasses
        raw = dict(obj)
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Enum):
            result[key] = value.value
        elif hasattr(value, "__dict__"):
            result[key] = _section_to_dict(value)
        else:
            result[key] = value
    return result


def parse_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """Parse CLI style ``key=value`` override pairs into a mapping."""

    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not in key=value format")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Override key must not be empty")
        overrides[key] = _coerce_value(raw_value.strip())
    return overrides


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false", "yes", "no", "on", "off"}:
        return _require_bool({"value": value}, "value")
    try:
        return _require_positive_int({"value": value}, "value")
    except ValueError:
        pass
    return value


def _profile_defaults() -> Dict[Profile, ProfileMetadata]:
    """Construct the built-in profile registry."""

    standard_defaults = {
        "oneiros": {
            "profile": "standard",
            "isa": "rv32im",
            "isa_mode": "standard",
            "zero_knowledge": False,
            "ivc_backend": "supernova",
            "recursion_curve": "pasta",
            "memory_model": "full_merkle",
            "page_size": 4096,
            "max_cycles_per_segment": 262144,
            "transcript": "poseidon2_v2",
            "security_bits": 100,
            "lookup_backend": "lasso",
            "hardware_backend": "cpu",
        },
        "proving": {
            "ldt": "deepfri",
            "stark_field": "m31",
            "pcs": "binius",
            "folding": "supernova",
        },
        "injector": {
            "wrapper": "plonky3",
            "primary_curve": "pallas",
            "fallback_curve": "vesta",
            "aggregation": "nova_fold",
            "blob_binding": "celestia_nmt",
        },
    }

    continuations_defaults = {
        "oneiros": {
            "profile": "continuations",
            "isa": "rv32im",
            "isa_mode": "per_instruction_air",
            "zero_knowledge": True,
            "ivc_backend": "hypernova",
            "recursion_curve": "pasta",
            "memory_model": "sparse_merkle_twist",
            "page_size": 4096,
            "max_cycles_per_segment": 1048576,
            "transcript": "poseidon2_v2",
            "security_bits": 128,
            "lookup_backend": "lasso_decomposable",
            "hardware_backend": "auto",
        },
        "proving": {
            "ldt": "deepfri",
            "stark_field": "m31",
            "pcs": "deepfold",
            "folding": "protogalaxy_optional",
        },
        "injector": {
            "wrapper": "groth16",
            "primary_curve": "bls12_381",
            "fallback_curve": "bn254",
            "aggregation": "snarkpack_cp_snark",
            "blob_binding": "eip4844_kzg",
        },
    }

    distributed_defaults = {
        "oneiros": {
            "profile": "distributed",
            "isa": "rv64im",
            "isa_mode": "per_instruction_air",
            "zero_knowledge": True,
            "ivc_backend": "hypernova_protogalaxy",
            "recursion_curve": "bls12_377",
            "memory_model": "sparse_merkle_twist",
            "page_size": 8192,
            "max_cycles_per_segment": 16777216,
            "transcript": "poseidon2_v2",
            "security_bits": 192,
            "lookup_backend": "caulk",
            "hardware_backend": "fpga",
        },
        "proving": {
            "ldt": "stir",
            "stark_field": "bls12_381_fr",
            "pcs": "deepfold",
            "folding": "hypernova",
        },
        "injector": {
            "wrapper": "supersonic",
            "primary_curve": "bls12_381",
            "fallback_curve": "bn254",
            "aggregation": "nova_fold",
            "blob_binding": "celestia_nmt",
        },
    }

    return {
        Profile.STANDARD: ProfileMetadata(
            profile=Profile.STANDARD,
            description="Single-segment developer profile optimised for CPU debugging.",
            defaults=standard_defaults,
            target_cycles=2_000_000,
            working_set_bytes=16 * 1024 * 1024,
            concurrency=1,
        ),
        Profile.CONTINUATIONS: ProfileMetadata(
            profile=Profile.CONTINUATIONS,
            description="Continuation-friendly profile with balanced proving throughput.",
            defaults=continuations_defaults,
            target_cycles=32_000_000,
            working_set_bytes=64 * 1024 * 1024,
            concurrency=4,
        ),
        Profile.DISTRIBUTED: ProfileMetadata(
            profile=Profile.DISTRIBUTED,
            description="High-throughput distributed profile targeting hardware accelerators.",
            defaults=distributed_defaults,
            target_cycles=128_000_000,
            working_set_bytes=256 * 1024 * 1024,
            concurrency=32,
        ),
    }


_PROFILE_REGISTRY = _profile_defaults()


__all__ = [
    "AppConfig",
    "ProfileMetadata",
    "InjectorConfig",
    "OneirosSection",
    "ProvingConfig",
    "available_profiles",
    "build_profile_config",
    "load_config",
    "profile_metadata",
    "parse_overrides",
]
