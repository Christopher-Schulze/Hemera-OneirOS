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
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Tuple


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
    ) -> None:
        self.page_size = page_size
        self.zero_knowledge = zero_knowledge
        self._pc = initial_pc
        self._cycle = 0
        self._registers: MutableMapping[int, int] = {i: 0 for i in range(32)}
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
            if not 0 <= rd < 32:
                raise ValueError("rd out of range")
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
    "TraceBuilder",
]

