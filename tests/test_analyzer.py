from oneiros.core.analyzer import analyze_register_activity, summarize_memory
from oneiros.core.trace import MemoryAccess, MemoryAccessType, TraceBuilder


def _build_sample_trace() -> TraceBuilder:
    builder = TraceBuilder(rng_seed=42)
    builder.emit_instruction("addi", rd=1, rs1=0, imm=1, result=1)
    builder.emit_instruction("addi", rd=2, rs1=1, imm=2, result=3)

    write_access = MemoryAccess(
        address=0x1000,
        value=0xDEADBEEF,
        size=4,
        access_type=MemoryAccessType.WRITE,
    )
    builder.emit_instruction("sw", rs1=1, rs2=2, imm=0, memory_accesses=[write_access])

    read_access = MemoryAccess(
        address=0x1000,
        value=0,
        size=4,
        access_type=MemoryAccessType.READ,
    )
    builder.emit_instruction("lw", rd=3, rs1=1, imm=0, result=0xDEADBEEF, memory_accesses=[read_access])

    exec_access = MemoryAccess(
        address=0x1000,
        value=0,
        size=4,
        access_type=MemoryAccessType.EXEC,
    )
    builder.emit_instruction("jal", rd=0, rs1=0, imm=4, next_pc=builder.snapshot_state().pc + 4, memory_accesses=[exec_access])
    return builder


def test_register_activity_analysis():
    builder = _build_sample_trace()
    bundle = builder.export()

    activities = analyze_register_activity(bundle, initial_registers={1: 0})
    regs = {activity.register: activity for activity in activities}

    assert regs[1].initial_value == 0
    assert regs[1].final_value == 1
    assert [event.value for event in regs[1].writes] == [1]

    assert regs[2].final_value == 3
    assert regs[3].final_value == 0xDEADBEEF

    all_registers = analyze_register_activity(
        bundle, initial_registers={1: 0}, include_untouched=True
    )
    assert len(all_registers) == 32


def test_memory_summary():
    builder = _build_sample_trace()
    summary = summarize_memory(builder.export())

    assert summary.writes == 1
    assert summary.reads == 1
    assert summary.execs == 1
    assert summary.copy_on_write_events == 1
    assert summary.unique_addresses == 1
    assert summary.unique_pages == 1
    assert summary.root_transitions == 0
    assert summary.final_root is not None
    assert summary.per_page
    page_stats = summary.per_page[0]
    assert page_stats.page_id == 1
    assert page_stats.writes == 1
    assert page_stats.reads == 1
    assert page_stats.execs == 1
    assert page_stats.unique_offsets == (0,)
