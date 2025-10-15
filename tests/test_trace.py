from __future__ import annotations

import re

import pytest

from oneiros.core.trace import (
    ChipType,
    MemoryAccess,
    MemoryAccessType,
    TraceBundle,
    TraceBuilder,
    TraceState,
)


def test_instruction_classification_defaults():
    builder = TraceBuilder(rng_seed=42)
    assert builder.classify_instruction("ADD") is ChipType.ALU
    assert builder.classify_instruction("jal") is ChipType.BRANCH
    assert builder.classify_instruction("lw") is ChipType.MEMORY
    assert builder.classify_instruction("mul") is ChipType.MUL
    assert builder.classify_instruction("poseidon2_batch") is ChipType.CUSTOM


def test_custom_instruction_registration():
    builder = TraceBuilder(rng_seed=1337)
    builder.register_instruction_category("FOO.BAR", ChipType.CUSTOM)
    assert builder.classify_instruction("foo.bar") is ChipType.CUSTOM


def test_emit_instruction_and_memory_rows():
    builder = TraceBuilder(rng_seed=1)

    # Write a value into memory
    write_access = MemoryAccess(
        address=0x1000,
        value=0xDEADBEEF,
        size=4,
        access_type=MemoryAccessType.WRITE,
    )
    cpu_row = builder.emit_instruction(
        "addi",
        rd=1,
        rs1=0,
        imm=5,
        result=5,
        memory_accesses=[write_access],
    )
    assert cpu_row.air_chip is ChipType.ALU
    assert cpu_row.registers_delta == ((1, 5),)
    assert cpu_row.zk_blind != 0

    bundle = builder.export()
    assert len(bundle.cpu) == 1
    assert len(bundle.memory) == 1
    mem_row = bundle.memory[0]
    assert mem_row.acc is MemoryAccessType.WRITE
    assert mem_row.cow is True  # first write triggers copy-on-write
    assert mem_row.val == 0xDEADBEEF
    assert mem_row.old_root != mem_row.new_root
    assert re.fullmatch(r"[0-9a-f]{64}", mem_row.new_root)

    # Read it back and confirm we observe the same value without COW
    read_access = MemoryAccess(
        address=0x1000,
        value=0,
        size=4,
        access_type=MemoryAccessType.READ,
    )
    builder.emit_instruction(
        "lw",
        rd=2,
        rs1=1,
        imm=0,
        result=0xDEADBEEF,
        memory_accesses=[read_access],
    )
    bundle = builder.export()
    assert len(bundle.cpu) == 2
    assert len(bundle.memory) == 2
    read_row = bundle.memory[-1]
    assert read_row.acc is MemoryAccessType.READ
    assert read_row.cow is False
    assert read_row.val == 0xDEADBEEF
    assert read_row.old_root == read_row.new_root


def test_unknown_instruction_raises():
    builder = TraceBuilder()
    with pytest.raises(ValueError):
        builder.emit_instruction("not_a_real_instr")


def test_trace_builder_snapshot_and_memory_preload():
    builder = TraceBuilder(rng_seed=99, initial_registers={5: 1337})
    builder.load_memory(0x2000, b"\xAA\xBB\xCC\xDD")
    snapshot = builder.snapshot_state()
    assert isinstance(snapshot, TraceState)
    assert snapshot.pc == 0
    registers = dict(snapshot.registers)
    assert registers[5] == 1337
    assert len(snapshot.memory_root) == 64

    builder.emit_instruction("addi", rd=5, rs1=0, imm=1, result=1338)
    second_snapshot = builder.snapshot_state()
    assert second_snapshot.memory_root == snapshot.memory_root
    bundle = builder.export()
    assert bundle.cpu[-1].result == 1338
    assert len(bundle.memory) == 0


def test_trace_builder_fork_and_rng_alignment():
    builder = TraceBuilder(rng_seed=7)
    builder.emit_instruction("addi", rd=1, rs1=0, imm=5, result=5)
    clone = builder.fork()

    builder.emit_instruction("addi", rd=3, rs1=0, imm=2, result=2)
    clone.emit_instruction("addi", rd=3, rs1=0, imm=2, result=2)

    orig_last = builder.export().cpu[-1]
    clone_last = clone.export().cpu[-1]
    assert orig_last.zk_blind == clone_last.zk_blind

    clone.emit_instruction("addi", rd=4, rs1=0, imm=1, result=1)
    assert len(clone.export().cpu) == len(builder.export().cpu) + 1


def test_trace_bundle_roundtrip_from_dict():
    builder = TraceBuilder(rng_seed=1)
    builder.emit_instruction("addi", rd=1, result=1)
    bundle = builder.export()
    payload = bundle.to_dict()
    reconstructed = TraceBundle.from_dict(payload)
    assert reconstructed == bundle

