"""Execution trace construction utilities for the OneirOS zkVM.

This module provides a high level, deterministic trace builder that mirrors
the normative CPU and memory trace schemas outlined in ``docs/zielbild``.  The
builder is intentionally lightweight: it does not attempt to emulate the full
guest ISA, yet it enforces enough structure to allow higher level tooling and
tests to reason about execution ordering, sparse memory updates and
zero-knowledge blinding.

The implementation focuses on three pillars:

* **CPU trace rows** closely following the spec's ``CPUTraceRow`` struct.  The
  builder automatically tracks the current ``pc`` and register deltas and
  derives chip classifications for common RISC-V, EVM and WASM mnemonics.
* **Sparse memory modelling** using copy-on-write 4 KiB pages.  Memory updates
  recompute deterministic digests so that successive rows expose realistic
  ``old_root``/``new_root`` transitions.
* **Convenient serialisation** helpers that export trace bundles as
  ``dict`` structures consumable by JSON/YAML pipelines or diagnostic CLIs.

The heuristics here are deliberately opinionated but deterministic: callers can
rely on the trace builder for reproducible tests and example pipelines without
having to hook into native proving backends.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple


# ---------------------------------------------------------------------------
#  Enumerations and dataclasses modelling the canonical trace representation
# ---------------------------------------------------------------------------


class ChipType(str, Enum):
    """Chip classification used to route instructions into AIR partitions."""

    ALU = "ALU"
    BRANCH = "BRANCH"
    MEMORY = "MEMORY"
    MUL = "MUL"
    CUSTOM = "CUSTOM"


class MemoryAccessType(str, Enum):
    """Types of memory accesses represented in ``MemTraceRow``."""

    READ = "READ"
    WRITE = "WRITE"
    EXEC = "EXEC"


@dataclass(frozen=True)
class SparseProof:
    """Placeholder sparse proof witness derived from memory commitments."""

    page_commitment: str
    witness_digest: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "page_commitment": self.page_commitment,
            "witness_digest": self.witness_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SparseProof":
        try:
            page_commitment = str(data["page_commitment"])
            witness_digest = str(data["witness_digest"])
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError("sparse proof mapping missing required keys") from exc
        return cls(page_commitment=page_commitment, witness_digest=witness_digest)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean value is not a valid integer")
    if isinstance(value, (int, float)) and int(value) == value:
        return int(value)
    if isinstance(value, str) and value.strip():
        cleaned = value.replace("_", "")
        return int(cleaned, 10)
    raise ValueError(f"invalid optional integer value: {value!r}")


@dataclass(frozen=True)
class MemoryAccess:
    """Request to read or write memory during a trace cycle."""

    address: int
    value: int
    size: int
    access_type: MemoryAccessType

    def __post_init__(self) -> None:
        if self.address < 0:
            raise ValueError("address must be non-negative")
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.value < 0:
            raise ValueError("value must be non-negative")


@dataclass(frozen=True)
class CPUTraceRow:
    """CPU execution trace entry as described in the specification."""

    cycle: int
    pc: int
    instr: str
    air_chip: ChipType
    rd: int | None
    rs1: int | None
    rs2: int | None
    imm: int
    result: int
    next_pc: int
    registers_delta: Tuple[Tuple[int, int], ...]
    flags: int
    zk_blind: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "cycle": self.cycle,
            "pc": self.pc,
            "instr": self.instr,
            "air_chip": self.air_chip.value,
            "rd": self.rd,
            "rs1": self.rs1,
            "rs2": self.rs2,
            "imm": self.imm,
            "result": self.result,
            "next_pc": self.next_pc,
            "registers_delta": list(self.registers_delta),
            "flags": self.flags,
            "zk_blind": self.zk_blind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CPUTraceRow":
        try:
            chip = ChipType(str(data["air_chip"]))
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError("cpu trace mapping missing 'air_chip'") from exc
        return cls(
            cycle=int(data.get("cycle", 0)),
            pc=int(data.get("pc", 0)),
            instr=str(data.get("instr", "")),
            air_chip=chip,
            rd=_optional_int(data.get("rd")),
            rs1=_optional_int(data.get("rs1")),
            rs2=_optional_int(data.get("rs2")),
            imm=int(data.get("imm", 0)),
            result=int(data.get("result", 0)),
            next_pc=int(data.get("next_pc", 0)),
            registers_delta=tuple(
                (int(reg), int(val)) for reg, val in data.get("registers_delta", [])
            ),
            flags=int(data.get("flags", 0)),
            zk_blind=int(data.get("zk_blind", 0)),
        )


@dataclass(frozen=True)
class MemTraceRow:
    """Memory trace entry produced for each load/store/exec access."""

    cycle: int
    addr: int
    page_id: int
    page_off: int
    acc: MemoryAccessType
    val: int
    sparse_proof: SparseProof
    old_root: str
    new_root: str
    cow: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "cycle": self.cycle,
            "addr": self.addr,
            "page_id": self.page_id,
            "page_off": self.page_off,
            "acc": self.acc.value,
            "val": self.val,
            "sparse_proof": self.sparse_proof.to_dict(),
            "old_root": self.old_root,
            "new_root": self.new_root,
            "cow": self.cow,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MemTraceRow":
        try:
            acc = MemoryAccessType(str(data["acc"]))
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError("memory trace mapping missing 'acc'") from exc
        sparse = data.get("sparse_proof", {})
        if not isinstance(sparse, Mapping):  # pragma: no cover - defensive guard
            raise ValueError("sparse_proof must be a mapping")
        proof = SparseProof.from_dict(sparse)
        return cls(
            cycle=int(data.get("cycle", 0)),
            addr=int(data.get("addr", 0)),
            page_id=int(data.get("page_id", 0)),
            page_off=int(data.get("page_off", 0)),
            acc=acc,
            val=int(data.get("val", 0)),
            sparse_proof=proof,
            old_root=str(data.get("old_root", "")),
            new_root=str(data.get("new_root", "")),
            cow=bool(data.get("cow", False)),
        )


@dataclass(frozen=True)
class TraceBundle:
    """Container bundling CPU and memory trace rows."""

    cpu: Tuple[CPUTraceRow, ...]
    memory: Tuple[MemTraceRow, ...]

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        return {
            "cpu": [row.to_dict() for row in self.cpu],
            "memory": [row.to_dict() for row in self.memory],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TraceBundle":
        cpu_rows = data.get("cpu", [])
        mem_rows = data.get("memory", [])
        if not isinstance(cpu_rows, Sequence) or not isinstance(mem_rows, Sequence):
            raise ValueError("Trace bundle data must contain sequence entries for cpu/memory")
        cpu = tuple(
            row if isinstance(row, CPUTraceRow) else CPUTraceRow.from_dict(row)
            for row in cpu_rows
        )
        memory = tuple(
            row if isinstance(row, MemTraceRow) else MemTraceRow.from_dict(row)
            for row in mem_rows
        )
        return cls(cpu=cpu, memory=memory)


@dataclass(frozen=True)
class TraceState:
    """Snapshot of the builder state at a given cycle."""

    cycle: int
    pc: int
    registers: Tuple[Tuple[int, int], ...]
    memory_root: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "cycle": self.cycle,
            "pc": self.pc,
            "registers": list(self.registers),
            "memory_root": self.memory_root,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TraceState":
        registers = tuple(
            (int(reg), int(val)) for reg, val in data.get("registers", [])
        )
        return cls(
            cycle=int(data.get("cycle", 0)),
            pc=int(data.get("pc", 0)),
            registers=registers,
            memory_root=str(data.get("memory_root", "")),
        )


# ---------------------------------------------------------------------------
#  Sparse memory model
# ---------------------------------------------------------------------------


class SparseMemoryImage:
    """Sparse, page-based memory image with deterministic commitments."""

    def __init__(self, page_size: int = 4096) -> None:
        if page_size <= 0 or page_size & (page_size - 1):
            raise ValueError("page_size must be a power-of-two positive integer")
        self.page_size = page_size
        self._pages: Dict[int, bytearray] = {}
        self._dirty_pages: set[int] = set()

    # -- page helpers -----------------------------------------------------

    def _ensure_page(self, page_id: int) -> bytearray:
        page = self._pages.get(page_id)
        if page is None:
            page = bytearray(self.page_size)
            self._pages[page_id] = page
        return page

    def _iter_chunks(self, address: int, size: int) -> Iterator[Tuple[int, int, int]]:
        remaining = size
        cursor = address
        while remaining > 0:
            page_id = cursor // self.page_size
            offset = cursor % self.page_size
            chunk = min(remaining, self.page_size - offset)
            yield page_id, offset, chunk
            cursor += chunk
            remaining -= chunk

    # -- operations -------------------------------------------------------

    def read(self, address: int, size: int) -> bytes:
        """Read ``size`` bytes starting from ``address``."""

        chunks: List[bytes] = []
        for page_id, offset, chunk in self._iter_chunks(address, size):
            page = self._pages.get(page_id)
            if page is None:
                chunks.append(bytes(chunk))
            else:
                chunks.append(bytes(page[offset : offset + chunk]))
        return b"".join(chunks)

    def write(self, address: int, data: bytes) -> bool:
        """Write ``data`` starting at ``address`` and return whether COW happened."""

        if not data:
            raise ValueError("data must not be empty")

        size = len(data)
        cursor = 0
        cow_triggered = False
        for page_id, offset, chunk in self._iter_chunks(address, size):
            page = self._ensure_page(page_id)
            if page_id not in self._dirty_pages:
                self._pages[page_id] = bytearray(page)  # copy-on-write snapshot
                self._dirty_pages.add(page_id)
                cow_triggered = True
                page = self._pages[page_id]
            page[offset : offset + chunk] = data[cursor : cursor + chunk]
            cursor += chunk
        return cow_triggered

    # -- commitments ------------------------------------------------------

    def page_commitment(self, page_id: int) -> str:
        """Return a deterministic digest for a single page."""

        page = self._pages.get(page_id)
        hasher = hashlib.blake2b(digest_size=32)
        if page is not None:
            hasher.update(page)
        return hasher.hexdigest()

    def root(self) -> str:
        """Return the aggregate memory root commitment."""

        hasher = hashlib.blake2b(digest_size=32)
        for page_id in sorted(self._pages):
            page_digest = hashlib.blake2b(self._pages[page_id], digest_size=32).digest()
            hasher.update(page_id.to_bytes(8, byteorder="little"))
            hasher.update(page_digest)
        return hasher.hexdigest()

    def opening_witness(self, page_id: int, offset: int, data: bytes) -> str:
        hasher = hashlib.blake2b(digest_size=32)
        hasher.update(page_id.to_bytes(8, "little"))
        hasher.update(offset.to_bytes(4, "little"))
        hasher.update(len(data).to_bytes(4, "little"))
        hasher.update(data)
        return hasher.hexdigest()

    def clone(self) -> "SparseMemoryImage":
        """Return a deep copy of the memory image."""

        clone = SparseMemoryImage(page_size=self.page_size)
        clone._pages = {page_id: bytearray(page) for page_id, page in self._pages.items()}
        clone._dirty_pages = set(self._dirty_pages)
        return clone


# ---------------------------------------------------------------------------
#  Instruction classification
# ---------------------------------------------------------------------------


_DEFAULT_INSTRUCTION_MAP: Mapping[str, ChipType] = {
    # ALU
    "add": ChipType.ALU,
    "addi": ChipType.ALU,
    "sub": ChipType.ALU,
    "and": ChipType.ALU,
    "andi": ChipType.ALU,
    "or": ChipType.ALU,
    "ori": ChipType.ALU,
    "xor": ChipType.ALU,
    "xori": ChipType.ALU,
    "sll": ChipType.ALU,
    "slli": ChipType.ALU,
    "srl": ChipType.ALU,
    "srli": ChipType.ALU,
    "sra": ChipType.ALU,
    "srai": ChipType.ALU,
    "slt": ChipType.ALU,
    "slti": ChipType.ALU,
    "sltu": ChipType.ALU,
    "sltiu": ChipType.ALU,
    "lui": ChipType.ALU,
    "auipc": ChipType.ALU,
    "not": ChipType.ALU,
    # Branch
    "beq": ChipType.BRANCH,
    "bne": ChipType.BRANCH,
    "blt": ChipType.BRANCH,
    "bge": ChipType.BRANCH,
    "bltu": ChipType.BRANCH,
    "bgeu": ChipType.BRANCH,
    "jal": ChipType.BRANCH,
    "jalr": ChipType.BRANCH,
    # Memory
    "lb": ChipType.MEMORY,
    "lbu": ChipType.MEMORY,
    "lh": ChipType.MEMORY,
    "lhu": ChipType.MEMORY,
    "lw": ChipType.MEMORY,
    "lwu": ChipType.MEMORY,
    "ld": ChipType.MEMORY,
    "sb": ChipType.MEMORY,
    "sh": ChipType.MEMORY,
    "sw": ChipType.MEMORY,
    "sd": ChipType.MEMORY,
    "prefetch": ChipType.MEMORY,
    # Multiplication / division
    "mul": ChipType.MUL,
    "mulh": ChipType.MUL,
    "mulhsu": ChipType.MUL,
    "mulhu": ChipType.MUL,
    "div": ChipType.MUL,
    "divu": ChipType.MUL,
    "rem": ChipType.MUL,
    "remu": ChipType.MUL,
    # Custom OneirOS extensions / crypto hints
    "poseidon2_batch": ChipType.CUSTOM,
    "commit_check": ChipType.CUSTOM,
    "lasso_lookup": ChipType.CUSTOM,
    "sparse_merkle_verify": ChipType.CUSTOM,
    "ecdsa_verify": ChipType.CUSTOM,
    "ed25519_verify": ChipType.CUSTOM,
    "bls_verify": ChipType.CUSTOM,
    "rsa_verify_2048": ChipType.CUSTOM,
    # Selected EVM / WASM mnemonics to ease frontend testing
    "keccak256": ChipType.CUSTOM,
    "mload": ChipType.MEMORY,
    "mstore": ChipType.MEMORY,
    "call": ChipType.CUSTOM,
    "ret": ChipType.BRANCH,
    "br_if": ChipType.BRANCH,
}


# ---------------------------------------------------------------------------
#  Trace Builder
# ---------------------------------------------------------------------------


class TraceBuilder:
    """Convenience builder that accumulates CPU and memory trace rows."""

    def __init__(
        self,
        *,
        page_size: int = 4096,
        zero_knowledge: bool = True,
        initial_pc: int = 0,
        rng_seed: int | None = None,
        initial_registers: Mapping[int, int] | None = None,
    ) -> None:
        self.page_size = page_size
        self.zero_knowledge = zero_knowledge
        self._pc = initial_pc
        self._cycle = 0
        self._registers: MutableMapping[int, int] = {i: 0 for i in range(32)}
        if initial_registers:
            for reg, value in initial_registers.items():
                self.set_register(reg, value)
        self._cpu_rows: List[CPUTraceRow] = []
        self._mem_rows: List[MemTraceRow] = []
        self._memory = SparseMemoryImage(page_size=page_size)
        self._instruction_map: Dict[str, ChipType] = dict(_DEFAULT_INSTRUCTION_MAP)
        self._rng = random.Random(rng_seed)
        self._current_root = self._memory.root()

    # -- public API -------------------------------------------------------

    def register_instruction_category(self, mnemonic: str, chip: ChipType) -> None:
        """Register or override the chip classification for ``mnemonic``."""

        if not mnemonic:
            raise ValueError("mnemonic must not be empty")
        self._instruction_map[mnemonic.strip().lower()] = chip

    def set_register(self, register: int, value: int) -> None:
        """Set the value for ``register`` in the builder state."""

        self._validate_register_index(register)
        self._registers[register] = int(value)

    def load_memory(self, address: int, data: bytes | bytearray) -> None:
        """Preload ``data`` into memory without emitting trace rows."""

        if address < 0:
            raise ValueError("address must be non-negative")
        if not data:
            raise ValueError("data payload must not be empty")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes-like")
        self._memory.write(address, bytes(data))
        self._current_root = self._memory.root()

    def snapshot_state(self) -> TraceState:
        """Return a snapshot of the current builder state."""

        registers = tuple(sorted(self._registers.items()))
        return TraceState(
            cycle=self._cycle,
            pc=self._pc,
            registers=registers,
            memory_root=self._current_root,
        )

    def fork(self) -> "TraceBuilder":
        """Create an independent clone of the builder including trace rows."""

        clone = TraceBuilder(
            page_size=self.page_size,
            zero_knowledge=self.zero_knowledge,
            initial_pc=self._pc,
            rng_seed=0,
        )
        clone._cycle = self._cycle
        clone._registers = dict(self._registers)
        clone._cpu_rows = list(self._cpu_rows)
        clone._mem_rows = list(self._mem_rows)
        clone._memory = self._memory.clone()
        clone._instruction_map = dict(self._instruction_map)
        clone._current_root = self._current_root
        clone._rng.setstate(self._rng.getstate())
        return clone

    def classify_instruction(self, mnemonic: str) -> ChipType:
        key = mnemonic.strip().lower()
        try:
            return self._instruction_map[key]
        except KeyError as exc:  # pragma: no cover - defensive mapping fallback
            raise ValueError(f"Unknown instruction mnemonic: {mnemonic}") from exc

    def emit_instruction(
        self,
        mnemonic: str,
        *,
        rd: int | None = None,
        rs1: int | None = None,
        rs2: int | None = None,
        imm: int = 0,
        result: int | None = None,
        next_pc: int | None = None,
        flags: int = 0,
        memory_accesses: Iterable[MemoryAccess] | None = None,
        custom_chip: ChipType | None = None,
    ) -> CPUTraceRow:
        """Append a CPU trace row (and optional memory rows)."""

        chip = custom_chip or self.classify_instruction(mnemonic)
        cycle = self._cycle
        pc = self._pc
        computed_next_pc = next_pc if next_pc is not None else pc + 4

        registers_delta: List[Tuple[int, int]] = []
        if rd is not None and result is not None:
            self._validate_register_index(rd)
            if self._registers.get(rd) != result:
                self._registers[rd] = result
                registers_delta.append((rd, result))

        registers_delta.sort(key=lambda item: item[0])
        zk_blind = self._next_blind()

        cpu_row = CPUTraceRow(
            cycle=cycle,
            pc=pc,
            instr=mnemonic,
            air_chip=chip,
            rd=rd,
            rs1=rs1,
            rs2=rs2,
            imm=imm,
            result=result if result is not None else 0,
            next_pc=computed_next_pc,
            registers_delta=tuple(registers_delta),
            flags=flags,
            zk_blind=zk_blind,
        )
        self._cpu_rows.append(cpu_row)

        if memory_accesses:
            for access in memory_accesses:
                self._mem_rows.append(self._process_memory_access(access, cycle))

        self._cycle += 1
        self._pc = computed_next_pc
        return cpu_row

    def export(self) -> TraceBundle:
        """Return an immutable bundle with all trace rows."""

        return TraceBundle(cpu=tuple(self._cpu_rows), memory=tuple(self._mem_rows))

    # -- internals --------------------------------------------------------

    def _next_blind(self) -> int:
        if not self.zero_knowledge:
            return 0
        # Use a 252-bit field-sized random for stability across runs.
        return self._rng.getrandbits(252)

    def _validate_register_index(self, register: int) -> None:
        if not 0 <= register < 32:
            raise ValueError("register index must be between 0 and 31")

    def _process_memory_access(self, access: MemoryAccess, cycle: int) -> MemTraceRow:
        address = access.address
        value = access.value
        size = access.size
        if size > 8:
            raise ValueError("Maximum supported access size is 8 bytes for tracing")
        if access.access_type == MemoryAccessType.WRITE:
            data = value.to_bytes(size, byteorder="little", signed=False)
            old_root = self._current_root
            cow = self._memory.write(address, data)
            new_root = self._memory.root()
            page_id = address // self.page_size
            page_off = address % self.page_size
            proof = SparseProof(
                page_commitment=self._memory.page_commitment(page_id),
                witness_digest=self._memory.opening_witness(page_id, page_off, data),
            )
        elif access.access_type in {MemoryAccessType.READ, MemoryAccessType.EXEC}:
            data = self._memory.read(address, size)
            old_root = self._current_root
            cow = False
            new_root = old_root
            page_id = address // self.page_size
            page_off = address % self.page_size
            proof = SparseProof(
                page_commitment=self._memory.page_commitment(page_id),
                witness_digest=self._memory.opening_witness(page_id, page_off, data),
            )
            value = int.from_bytes(data, byteorder="little", signed=False)
        else:  # pragma: no cover - Enum exhaustiveness guard
            raise ValueError(f"Unhandled access type: {access.access_type}")

        self._current_root = new_root
        return MemTraceRow(
            cycle=cycle,
            addr=address,
            page_id=page_id,
            page_off=address % self.page_size,
            acc=access.access_type,
            val=value,
            sparse_proof=proof,
            old_root=old_root,
            new_root=new_root,
            cow=cow,
        )


__all__ = [
    "ChipType",
    "MemoryAccessType",
    "SparseProof",
    "MemoryAccess",
    "CPUTraceRow",
    "MemTraceRow",
    "TraceBundle",
    "TraceState",
    "TraceBuilder",
]

