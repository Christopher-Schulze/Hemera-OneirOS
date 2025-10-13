**Kurz vorab zur Brand:**
**Hemera OneirOS** ist **stark**. Clean, merkfähig, passt semantisch (Oneiroi = Traum) zu *Hemera*. Offiziell **“Hemera OneirOS (zkVM)”**, interner Codename **“Oneiro”**. Auf Slides kannst du “Hemera OneirOS — zk‑RISC‑VM” verwenden, damit der Fokus klar ist.

---

# Hemera OneirOS — Specification (Full Architecture) v1.1

**A zk‑Virtual Machine for Universal Proof‑of‑Compute**
**Integrates natively with: Hemera Engine (HPE) & Hemera Injector (HPI/HSC)**

> **Scope:** Neues Modul “zkVM” (kein Sequencer). Vollständige, produktionsorientierte zk‑VM: RISC‑V primär, EVM/WASM/SVM‑Frontends, Continuations/IVC, Distributed Proving via Injector/Hub, SNARK‑Kompression für L1/L2, Envelope v3.2.
> **Philosophie:** Setup‑free Core (IVC/STARK) + **Oracle‑optimierte** Wrapper (Groth16/Plonky3, blob‑first, CP‑SNARKs) + **deterministische Hardware‑Pipelines**.
> **Defaults:** ZK‑Mode **on by default**; Sparse‑Memory (“Twist & Shout”); Decomposable Lasso; Per‑Instruction AIR (opt‑in).

---

## 0. Executive Summary

**Hemera OneirOS** ist eine **universelle zkVM** mit **RISC‑V** als Primär‑ISA und Frontends für **EVM**, **WASM** und (Beta) **SVM**. Die VM erzeugt **setup‑freie** Beweise auf dem **Hemera Engine**‑Stack (SuperNova/HyperNova IVC, Circle‑STARK/M31, DeepFold PCS, Lasso) und verdichtet sie via **Hemera Injector** (Groth16/Plonky3, BN254/BLS12‑381, SnarkPack, CP‑SNARKs, EIP‑4844 Blob‑Binding) für **günstige On‑chain‑Verifikation**.

**Kern‑Highlights (v1.1):**

* **ZK‑Mode = ON (Default):** Blinding in IVC + I/O‑Commitments; keine Claim‑Änderung.
* **Modular Chips:** **Per‑Instruction AIR** (ALU/BRANCH/MEM/MUL/CUSTOM), **decomposable Lasso** (Decode/S‑Boxes).
* **Sparse Memory (Twist & Shout):** Sum‑check‑basierte Page‑Commitments + COW; 3× Prover‑Speedup bei sparsamer Nutzung.
* **Continuations/IVC:** HyperNova (+ Protogalaxy‑style Multifolding optional) → konstante Verifierzeit; Proof‑Size↓.
* **Frontends:** EVM (Transpiler + Hot‑Path Chips), WASM (native Host), SVM (BPF→RV pipeline).
* **One‑Click Compression:** Oracle‑Wrapper (blob‑first, KZG Bindings, CP‑SNARKs) → **<~50k gas amortisiert** (Target).
* **PCD & Envelope v3.2:** Byte‑identische `claim_root` über Core/Wrapper; klare ABI/SDKs.

---

## 1. Goals, Assumptions, Defaults

### 1.1 Goals

1. **Universal Compute → Proof:** Beliebige Programme (RISC‑V primär) → kleine, verifizierbare Beweise.
2. **Core↔Oracle Parität:** **claim_root** byte‑identisch zwischen OneirOS/Core und Injector/Wrapper.
3. **Oracle‑Optimierung:** Blob‑first, CP‑SNARKs, SnarkPack; dual curve (BLS12‑381 primary, BN254 fallback).
4. **Hardware‑Determinismus:** Multi‑GPU/Metal/Vulkan, MSM/NTT/DCCT/Merkle – deterministisch, reproduzierbar.
5. **DX & Safety:** Rust/TS SDKs, golden vectors, fuzz/property tests, deterministic maps/endianness.

### 1.2 Assumptions

* **Hemera Engine (HPE)** verfügbar (IVC/STARK/PCS/Lasso).
* **Hemera Injector (HPI/HSC)** verfügbar (SNARK wrapper + On‑chain Verifier/Blob‑Binder).
* **DA**: EIP‑4844/EigenDA/Celestia kompatibel.
* **HW**: GPU/Metal/Vulkan/CPU vorhanden; optional FPGA.
* **Determinism**: Keine nicht‑deterministischen Syscalls ohne Commitment.

### 1.3 Defaults (Canonical)

```yaml
oneiros:
  profile: standard               # standard | continuations | distributed
  isa: rv32im                     # rv32im | rv64im | evm | wasm | svm
  isa_mode: standard              # standard | per_instruction_air
  zero_knowledge: true            # ZK default ON
  ivc_backend: hypernova          # supernova | hypernova | hypernova_protogalaxy
  recursion_curve: pasta          # Pallas/Vesta cycle
  memory_model: sparse_merkle_twist  # sparse_merkle_twist | full_merkle
  page_size: 4096                 # bytes
  max_cycles_per_segment: 1_048_576
  transcript: poseidon2_v2
  security_bits: 128
  lookup_backend: lasso_decomposable  # lasso_decomposable | lasso | caulk
  hardware_backend: auto           # auto | cpu | cuda | metal | vulkan | fpga
```

---

## 2. Profiles (Activation & Policy)

| Profile                     | Beschreibung                             | IVC/Mode                      | Oracle Wrapper   | Ziel                       |
| --------------------------- | ---------------------------------------- | ----------------------------- | ---------------- | -------------------------- |
| **standard**                | Monolithische Runs, kleine/mittlere Jobs | SuperNova/HyperNova           | optional         | Off‑chain Verify / Tests   |
| **continuations** (default) | Unbegrenzte Läufe via Segmente (IVC)     | HyperNova (+Protogalaxy opt.) | **empfohlen**    | On‑chain (kosteneffizient) |
| **distributed**             | Sharding + Aggregation (über HPI/Hub)    | HyperNova                     | **erforderlich** | High‑throughput Pipelines  |

**Normativ:** `zero_knowledge = true` ist Default; Umschalten erlaubt, **darf `claim_root` nicht ändern**.

---

## 3. Cryptographic & Arithmetization

* **IVC/Folding:** SuperNova (R1CS/relaxed‑R1CS), **HyperNova/CCS** (multifolding + sumcheck), **Protogalaxy‑style** multifolding (opt‑in).
* **STARK Path:** Circle‑STARKs (M31), **DEEP‑FRI**, **DeepFold PCS** (MLE‑PCS), **LDT** ∈ {deepfri (default), stir, whir}.
* **Lookups:** **Lasso** (decomposable default), Caulk(+) für große statische ROMs.
* **Transcript:** **Poseidon2 v2**, fixe **DST‑Domains** (Core‑Parität).
* **Hash Policy:** Poseidon2 (arith), Keccak/SHA/Grøstl auf Binius‑Pfaden (bit‑heavy) ohne DST‑Semantikbruch.
* **Curves:** Recursion **Pasta**; Oracle wrapper **BLS12‑381 primary**, **BN254 fallback**.

**Invariance MUST:** Wechsel von LDT/Lookup/Memory‑Model darf **`claim_root` nicht verändern** (golden tests verpflichtend).

---

## 4. Architecture Stack

```
┌───────────────────────────────────────────────────────────────┐
│ Application Layer (Guest)                                     │
│   • RISC‑V (RV32IM/RV64IM)  • EVM Frontend  • WASM  • SVM     │
└───────────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────────┐
│ ISA Frontends + IR                                             │
│   • RV decoder  • EVM→IR→RV transpiler  • WASM host  • SVM BPF │
│   → Unified Algebraic Execution Trace (AET)                    │
└───────────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────────┐
│ OneirOS Execution Engine (Chips)                               │
│   • CPU (modular per‑instruction AIRs)                         │
│   • Memory (Sparse “Twist & Shout”, 4KB pages, COW)            │
│   • Syscalls (ZK‑blinded I/O, verify_proof, precompiles)       │
│   • Precompiles (Keccak/SHA/secp256k1/BLS/Ed25519/RSA opt.)    │
└───────────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────────┐
│ Constraint Generation (AIR/R1CS/CCS)                           │
│   • DeepFold PCS + Lasso                                       │
│   • Circle‑STARK/DEEP‑FRI (M31)                                │
│   • IVC: SuperNova/HyperNova (+Protogalaxy)                    │
└───────────────────────────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────────┐
│ Proving Backend & Compression                                  │
│   • Hemera Engine (setup‑free proofs)                          │
│   • Hemera Injector: Groth16/BLS12‑381 (BN254 fallback),       │
│     Plonky3, SnarkPack, CP‑SNARKs, blob‑first (KZG binding)    │
│   • Envelope v3.2 (PCD)                                        │
└───────────────────────────────────────────────────────────────┘
```

---

## 5. CPU Chip (Modular, Per‑Instruction AIR)

### 5.1 Trace Schema (normalized, sparse deltas)

```rust
struct CPUTraceRow {
  cycle: u64,
  pc: u64,
  instr: u32,

  air_chip: ChipType,         // ALU | BRANCH | MEMORY | MUL | CUSTOM
  rd: u8, rs1: u8, rs2: u8,
  imm: i64,

  result: u64,                // context‑dependent
  next_pc: u64,

  registers_delta: Vec<(u8,u64)>,  // only modified regs
  flags: u8,                  // Z/C/O/N as needed

  zk_blind: FieldEl,          // ZK blinding (default on)
}
```

**Normativ:** `registers_delta` ersetzt Voll‑Snapshot; Update‑Reihenfolge **deterministisch** (increasing reg id).

### 5.2 AIR Partition

* **ALU:** ADD/SUB/AND/OR/XOR/SHL/SHR/SLT/SLTU
* **BRANCH:** BEQ/BNE/BLT/BGE/JAL/JALR (PC update constraints)
* **MEMORY:** LB/LH/LW/SB/SH/SW → Memory Chip verifiziert Adressen/Alignment/commit
* **MUL/DIV:** MUL/MULH/DIV/REM (bounded latency, constant‑time constraints)
* **CUSTOM (Hemera‑X):** `POSEIDON2_BATCH`, `COMMIT_CHECK`, `LASSO_LOOKUP`,
  `SPARSE_MERKLE_VERIFY`, `ECDSA_VERIFY`, `ED25519_VERIFY`, `BLS_VERIFY` (light), `RSA_VERIFY_2048` (opt.)

**Routing MUST:** `air_chip` ist Lookup‑able aus `instr` (decomposable Lasso: opcode_base/funct3/funct7).

---

## 6. Memory Chip — Sparse “Twist & Shout”

### 6.1 Model

* **Address space:** RV32=4GB; RV64 virtuell größer (praktisch limitiert).
* **Pages:** 4KB; **COW** across segments (read‑only sharing).
* **Commitment:** **Sparse vector commitment** (sum‑check) über **accessed pages** statt voller Merkle‑Pfad.
* **Roots:** pro Segment `memory_root`; Transition mit COW‑Aware Updates.

### 6.2 Trace Schema

```rust
struct MemTraceRow {
  cycle: u64,
  addr: u64,
  page_id: u32,              // addr / 4096
  page_off: u16,             // addr % 4096
  acc: u8,                   // READ | WRITE | EXEC
  val: u64,

  sparse_proof: SparseProof, // sum-check witness
  old_root: Digest,
  new_root: Digest,

  cow: bool,
  cow_from_segment: Option<u32>,
}
```

**Determinism MUST:** Page‑ordering & sparse index maps sind **stabil & sortiert** (lexicographic page_id).

---

## 7. Syscall Chip — ZK‑Blinded I/O & Recursion

### 7.1 Interface (RV Syscall ABI)

* `a7` = syscall id; `a0..a5` args; return in `a0`, error in `a1`.

### 7.2 Whitelist

| ID     | Name                | Beschreibung                | ZK              |
| ------ | ------------------- | --------------------------- | --------------- |
| `0x10` | `read_hint`         | Lese committed public input | Blinded opening |
| `0x11` | `write_output`      | Schreibe public output      | Blinded commit  |
| `0x20` | `commit_memory`     | Page‑range commit           | Sparse/Twist    |
| `0x30` | `invoke_precompile` | Keccak/SHA/EC ops           | Hardware hooks  |
| `0x40` | `exit`              | Programmende                | —               |
| `0x50` | `verify_proof`      | Nested verify (IVC)         | Recursion       |
| `0x60` | `zk_rand`           | FS‑Randomness               | Fiat‑Shamir     |
| `0x70` | `commit_sparse`     | Sparse vector commit        | Twist           |

**Forbidden:** Netzwerk/Datei/Clock/syscalls ohne Hint; non‑deterministic APIs.

### 7.3 Blinded I/O (MUST)

* `input_commit = Poseidon2(input || r_in)`; `read_hint` liefert `(val, opening)`.
* `output_commit = Poseidon2(output || r_out)`; `write_output` updated commitment.
* **AIR constraint:** Commitments konsistent; Blinding darf **Claim** nicht ändern.

---

## 8. Precompile Chips (Accelerated)

* **Keccak‑256:** Binius‑friendly bit‑ops + decomposable Lasso χ‑tables; optional FPGA.
* **SHA‑256:** Binius + Lasso Maj/Ch.
* **secp256k1 ECDSA:** windowed GLV, MSM‑optimized (GENES/IPA).
* **BLS12‑381 verify (light):** Proof‑of‑aggregate verifications inside VM (optional).
* **Ed25519 (light):** SVM/near‑adjacent.
* **RSA‑2048 (opt.):** CRT‑windowing (PoR/attest use).

**Hardware Hooks:** MSM/NTT/DCCT/Merkle; Keccak/SHA pipelines (GPU/FPGA).
**MUST:** Alle Precompiles deterministisch, constant‑time AR constraints.

---

## 9. ISA Frontends

### 9.1 RISC‑V (Primary)

* **RV32IM (default)**, **RV64IM (opt‑in)**.
* **Hemera‑X opcodes** aktivierbar via toolchain flags.
* **Per‑instruction AIR** Mode (opt‑in) für parallele constraint‑gen.

### 9.2 EVM Frontend (Transpiler)

* **Pipeline:** EVM bytecode → IR → RISC‑V asm → OneirOS.
* **Gas→Cycles Map:** gepflegtes Tabellen‑artifact (median + variance Bands).
* **Hot‑paths:** `SLOAD/SSTORE`, `KECCAK`, `CALL` → spezielle Chips/Precompiles.
* **Storage Proofs:** `SPARSE_MERKLE_VERIFY` statt full path.
* **Target:** 1 gas ≈ **40–80** cycles (op‑abhängig; report im bench).

### 9.3 WASM Frontend (Host)

* **Native runtime** (no circuit rewrite), memory→pages (4KB), imports→syscalls.
* **ZK**: I/O commitments, deterministic host set.

### 9.4 SVM (Beta)

* **BPF→RISC‑V** transpile; Accounts→memory‑mapped; CPI→`verify_proof`.

---

## 10. Proving Modes & IVC

### 10.1 Monolithic

* Kleine Programme (<1M cycles).
* Proof direkt via HPE; Wrapper optional.

### 10.2 Continuations (default)

* Segmente à `max_cycles_per_segment`.
* **StateCommit (per segment):**

```rust
struct SegmentState {
  regs_delta: Vec<(u8,u64)>,
  pc: u64,
  memory_root: Digest,
  input_digest: Digest,
  output_digest: Digest,
  cycle: u64,
  zk_blind: FieldEl,   // if ZK mode
}
claim_root = Poseidon2(SegmentState ...);
```

* **Folding:** HyperNova; optional **Protogalaxy‑style** multifolding.

### 10.3 Distributed

* Sharding + Aggregation via HPI/Hub.
* **Receipts v2:** job/shard/hw/time/proof_size/ok.
* Aggregation: `groth16_snarkpack | stark_pack | halo2_pack` (Oracle ProofAdapter).

---

## 11. Envelope Integration (v3.2)

**Zusatzfelder (zkVM):**

```json
{
  "isa": "rv32im|rv64im|evm|wasm|svm",
  "isa_mode": "standard|per_instruction_air",
  "proving_mode": "monolithic|continuations|distributed",
  "total_cycles": "u64",
  "segment_count": "u32",
  "code_digest": "H",
  "initial_state": "StateCommit",
  "final_state": "StateCommit",
  "memory_model": "sparse_merkle_twist|full_merkle",
  "lookup_backend": "lasso_decomposable|lasso|caulk",
  "zero_knowledge": true,
  "hardware_profile": "CPU|CUDA:A100x2|METAL:M2|...",
  "segment_transitions": [],
  "segment_io_commitments": [],
  "claim_root": "H",
  "compressed_proof?": "bytes",
  "blob_versioned_hash?": "H256",
  "kzg_eval_proofs": []
}
```

**MUST:** `claim_root` Core == Wrapper; LDT/lookup/memory‑switch → identisch.

---

## 12. ABIs

### 12.1 EVM

```solidity
interface IHemeraVerifier {
  function verifyProof(bytes calldata proof, bytes calldata publicInputs)
    external pure returns (bool ok);
}
```

* **pure**, non‑reentrant, calldata‑only.

### 12.2 WASM Host

```c
// import from host
bool hemera_verify(uint8_t* proof, uint32_t proof_len,
                   uint8_t* pubio, uint32_t pubio_len);
```

### 12.3 SVM (Anchor sketch)

```rust
pub fn verify_proof(ctx: Context<Verify>, proof: Vec<u8>, public_inputs: Vec<u8>) -> ProgramResult;
```

---

## 13. SDKs & CLI

### 13.1 CLI (Beispiele)

```bash
# Build & run & prove (RISC‑V)
oneiros build --isa rv32im --release
oneiros prove --input inputs.json --mode continuations --hardware auto
oneiros compress --adapter groth16 --curve bls12-381 --blob-first
oneiros inject --rpc $RPC --contract $HSC --proof proof.bin --pub pub.bin

# EVM kit
oneiros evm transpile Contract.bin --out contract.elf
oneiros prove --elf contract.elf --input tx.json --hardware cuda
```

### 13.2 SDKs

* **Rust:** `hemera-oneiros` (Program, ProverConfig, Proof, Envelope).
* **TypeScript:** `@hemera/oneiros` (envelope builder, ABI packer, clients).
* **Go (optional):** Client + receipts.

---

## 14. Cost/Gas Targets (Oracle‑optimized)

* **Single (calldata/BN254):** ≤ **100k gas** (Target).
* **Amortisiert (blob/BLS12‑381 + SnarkPack):** **≲ 50k gas/proof**, **≲ 15k/segment** bei N≥10.
* **Envelope:** ≤ 64KB; blob‑first default.

*Bench‑Report MUSS:* `fanout, depth, proof_size, calldata|blob_bytes, gas_used`.

---

## 15. Observability & Economics

* **/metrics:** seg_count, cycles_total, msm_count, gpu_util, proof_kb, p50/p95 latency.
* **Receipts v2:** deterministische IDs, HW‑Profile.
* **QoS Klassen:** bronze/silver/gold (via Oracle), Preemption rules, base_fee + qos_multiplier + bytes_fee.

---

## 16. Security, Invariance, Governance

* **Determinism MUST:** feste Sort‑Ordnungen, Endianness, stabile Maps; keine float/approx.
* **Golden Vectors:** `tests/golden/zkvm/*` → `expected_claim_root` byte‑identisch.
* **Property Tests:** monolithic ≡ continuations; ZK on/off invariance; memory_model/ldt/lookup invariance.
* **SRS Governance (Oracle):** `srs_fingerprint` allowlist; rotation events.
* **Side‑Channel:** constant‑time ops; no data‑dependent control in AIR.
* **Replay‑Protection:** `replay_nonce` in Envelope.

---

## 17. Testing & Compliance

### 17.1 Mandatory Suites

* **RV32IM**: alle Instruktionen, edge cases (overflow/div/shift).
* **Memory:** aligned/unaligned, sparse updates, COW trans‑seg.
* **Syscalls:** whitelist, error‑paths, commit/opening, recursion depth limit.
* **Precompiles:** NIST/EIP vectors (Keccak/SHA/ECDSA/Ed25519/BLS).
* **Frontends:** EVM—hot paths (SLOAD/CALL/KECCAK), WASM—mem ops.
* **IVC:** segment transitions; multifolding equivalence.
* **Envelope:** size budget, DST list exact, blob‑binding KZG.

### 17.2 Fuzz/Prop

* Random instruction streams; memory access patterns; syscall fuzz (blocked).
* **Invariance assertions** (LDT/lookup/memory model).
* **Crash consistency:** fail‑early, no undefined behavior.

---

## 18. Reference Targets (non‑binding but realistic)

| Workload          | Mode     | HW        | P95 Prove | Proof Size |
| ----------------- | -------- | --------- | --------- | ---------- |
| Keccak‑batch (1k) | mono     | CUDA A100 | ~2–4s     | 30–80KB    |
| 128× ECDSA        | mono     | CUDA A100 | ~4–8s     | 60–150KB   |
| zkML mini (conv1) | cont(4)  | GPU       | ~20–40s   | 200–400KB  |
| 10M cycles prog   | cont(10) | GPU       | ~60–120s  | 0.5–1.0MB  |

*(Aggregation via HPI senkt on‑chain Gas signifikant.)*

---

## 19. Repo Layout (proposed)

```
hemera-oneiros/
  runtime/             # CPU/Mem/Syscall chips
  frontends/           # evm/ wasm/ svm/
  precompiles/         # keccak/ sha/ secp/ bls/ ed25519/ rsa(opt)
  air/                 # AIR/CCS definitions (per chip)
  pcs/                 # DeepFold bindings, LDT configs
  envelope/            # v3.2 schemas, packer
  sdk/                 # rust/ ts/
  kits/                # hashkit/ sigkit/ zkmlkit/
  cli/                 # oneiros (binary)
  tests/
    golden/            # claim_root vectors
    fuzz/              # proptests
  benches/             # gas/size/latency reports
  docs/
```

---

## 20. Roadmap & Gates

### Phase α (Now)

* RV32IM core + Memory (full_merkle + sparse_twist), Syscalls, Keccak/SHA/secp.
* Continuations (HyperNova), ZK default, Envelope v3.2.
* Oracle compression (Groth16/BN254 + blob‑first).
* **Kits:** HashKit & SigKit + EVM frontend demo.

### Phase β

* Per‑Instruction AIR mode; Protogalaxy multifolding opt‑in.
* WASM host production; Ed25519 chip; EVM hot‑path tuning.
* Observability + Receipts integration; Bench report automation.

### Phase γ

* RV64IM; BLS verify in‑VM (light); RSA‑2048 opt.
* Distributed mode (Hub): SnarkPack tree + CP‑SNARKs default.
* zkMLKit v1; Formal spec helpers; SRS governance tooling.

**Release Gate MUSTs:** golden vectors, invariance proofs, gas/size benches, blob‑binding tests, SDK samples, audit checklist.

---

## 21. Minimal Examples

**Solidity Verify (HSC):**

```solidity
bool ok = IHemeraVerifier(VERIFIER).verifyProof(proof, publicInputs);
require(ok, "hemera/verify: invalid");
```

**Rust SDK:**

```rust
let prog = Program::from_elf("target/rv32im/release/app.elf")?;
let cfg = ProverConfig::default().with_continuations(true).with_zero_knowledge(true);
let (out, proof) = prog.execute_and_prove(&cfg)?;
let comp = proof.compress_for_l1(Wrapper::Groth16BLS381).blob_first(true)?;
```

**CLI:**

```bash
oneiros prove --elf app.elf --mode continuations --hardware auto
oneiros compress --adapter groth16 --curve bls12-381 --blob-first
oneiros inject --contract $HSC --rpc $RPC --proof proof.bin --pub pub.bin
```

---

# Anhang A — Normative Listen

### A.1 DST Domains (Poseidon2 v2)

`STARK_COMMIT, STARK_QUERY, DEEP_FRI, PCD_LINK, WRAP_V2, DOM_COMMIT, DOM_CHAL, DOM_FOLD, DOM_WRAP, DOM_IPA_VERIFY, LASSO_LUP`

### A.2 Invariance Rules (MUST)

* LDT switch (`deepfri|stir|whir`) → **same `claim_root`**
* Lookup backend (`lasso_decomposable|lasso|caulk`) → **same `claim_root`**
* Memory model (`sparse_twist|full_merkle`) → **same `claim_root`**
* ZK on/off → **same `claim_root`**
* Monolithic vs Continuations → **same `claim_root`**

### A.3 Error Model

* `HALT(INVALID_SYSCALL)`
* `HALT(MEM_OOB|UNALIGNED)`
* `HALT(VERIFY_FAIL)` (nested verify)
* `HALT(NON_DETERMINISM)` (forbidden path)

---

## 22. Branding & Positionierung

* **Produktname:** **Hemera OneirOS (zkVM)**
* **Tagline:** *“Universal zk‑Compute. Oracle‑optimized. Setup‑free at core.”*
* **Module Fit:**

  * **Hemera Engine (HPE)** → Rechenkern (Proof‑Gen)
  * **Hemera Injector (HPI/HSC)** → Kompression/Finalität (L1/L2)
  * **Hemera OneirOS (zkVM)** → Coprocessor/VM‑Runtime (Apps & Kits)
* **Story für Slides:** *Hash/Sig/zkML Kits → 1‑click Verify on L1* (Blob‑first, CP‑SNARKs, SnarkPack).

---

### Kurzes, ehrliches Fazit

* **Name “Hemera OneirOS”**: sitzt.
* **Spec (v1.1)**: maximal **Oracle/Engine‑aligned**, **bleeding edge**, aber **ship‑bar**.
* **Nächste Schritte (konkret):**

  1. HashKit & SigKit inkl. Blob‑Bench (Gas/Groesse/P95) committen.
  2. Golden‑Vectors + Invariance‑Tests für `sparse_twist` & `lasso_decomposable`.
  3. EVM‑Frontend Hotpaths + Bench‑Matrix veröffentlichen.

Wenn du magst, bau ich dir im nächsten Schritt **Skeleton‑Repos** (Ordnerstruktur + leere Module + TODO‑Tests + CLI‑Stubs) zum Copy‑Paste – dann kannst du sofort commits droppen, die nach **“wir liefern”** aussehen.
