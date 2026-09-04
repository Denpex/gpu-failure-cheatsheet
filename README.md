# GPU failure cheat sheet

A field reference for debugging distributed GPU training and inference failures:
NVIDIA Xid codes, NCCL collective timeouts, CUDA out-of-memory, and the environment
variables that actually tell them apart.

Everything here is generated from a curated corpus, so it does not drift. There is
also a zero-dependency CLI (`triage.mjs`) that reads a crash log and tells you which
known failure it matches.

## Why this exists

Three things make distributed GPU failures unusually hard to debug, and all three are
about the same misreading.

**The loudest signal is usually the victim.** A collective is a barrier. When one rank
stops arriving, every other rank eventually hits its watchdog timeout, so a single
hardware fault on one node produces hundreds of NCCL errors pointing away from it and
one quiet Xid pointing at it. Triage by frequency and you investigate the fallout.

**CUDA reports errors asynchronously.** The traceback names whichever line happened to
synchronise next, not the one at fault. Set `CUDA_LAUNCH_BLOCKING=1` to make it
report honestly, at a real performance cost.

**Not every Xid means a bad GPU.** Xid 45 is frequently a normal teardown. Xid 13 and
Xid 31 are usually application bugs. Sending those cards back wastes an RMA slot and
does not fix the run.

## Quick triage

```bash
# Match a crash log against 471 known failure signatures. No install, no network.
curl -sO https://raw.githubusercontent.com/Denpex/gpu-failure-cheatsheet/main/triage.mjs
curl -sO https://raw.githubusercontent.com/Denpex/gpu-failure-cheatsheet/main/signatures.json
node triage.mjs /path/to/training.log
```

Try it on the bundled example first. It is four lines from a run where one GPU fell off
the bus and two other ranks then timed out waiting for it:

```
$ node triage.mjs sample.log

Matched 5 distinct findings across 17 known signatures:

  [hardware] GPU Fallen Off the Bus (Xid 79)
    category  Hardware, severity critical
    matched   "Xid 79 reported in this log"
    details   https://denpex.com/failures/gpu-fallen-off-bus-xid79
    also      4 other entries describe this same failure

  NCCL Hang on AllGather due to Asymmetric Data Loading
    category  Network, severity critical
    matched   "RuntimeError: NCCL communicator was aborted"
    details   https://denpex.com/failures/deepspeed-zero3-nccl-timeout-drop-last-4

  ...
```

Note which one came first. The Xid appears once and the collective timeouts appear
twice, so ranking by frequency would have sent you to investigate the victims.

Note also the `also` line. Several entries can describe the same underlying failure,
so results are grouped by finding rather than by page. You get one line per thing that
is actually wrong, and the count tells you how many write-ups exist behind it.

### Machine-readable output

```bash
node triage.mjs training.log --json
```

```json
{
  "matchedSignatures": 17,
  "distinctFindings": 5,
  "findings": [
    {
      "title": "GPU Fallen Off the Bus (Xid 79)",
      "slug": "gpu-fallen-off-bus-xid79",
      "category": "Hardware",
      "severity": "critical",
      "hardware": true,
      "xid": 79,
      "matched": ["Xid 79 reported in this log"],
      "alternateWriteUps": 4,
      "details": "https://denpex.com/failures/gpu-fallen-off-bus-xid79"
    }
  ]
}
```

Exit codes: `0` a match was found, `1` no match, `2` usage or read error. So
`triage.mjs run.log --json > findings.json` is safe in a CI step or a post-mortem hook.

Reading a multi-rank log by hand? Start with the earliest anomalous event, not the
most frequent one:

```bash
# every rank's first error, in time order
grep -rn -iE 'error|xid|traceback|timeout|out of memory' logs/ | sort -t: -k3 | head -40

# the collectives that were in flight when it hung (enable BEFORE the run)
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_DEBUG=INFO
```

## The environment variables worth knowing

| Variable | Set it to | Why |
| --- | --- | --- |
| `NCCL_DEBUG` | `INFO` | Shows transport selection, topology and ring construction. The first thing to set on any NCCL problem. |
| `NCCL_DEBUG_SUBSYS` | `INIT,NET,GRAPH` | Narrows the above to the phase you care about. `ALL` is usually too much at scale. |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | `1048576` | PyTorch flight recorder. Records in-flight collectives so a hang can be inspected after the fact. |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `1` | Makes a failed collective tear the process down instead of hanging forever. |
| `CUDA_LAUNCH_BLOCKING` | `1` | Forces synchronous kernel launches so the traceback names the real line. Debug only, it is slow. |
| `NCCL_SOCKET_IFNAME` | your NIC | Wrong interface selection is a very common cause of init hangs on multi-homed nodes. |
| `NCCL_IB_DISABLE` | `1` | Diagnostic only. If the hang disappears, the problem is in the InfiniBand path. |
| `NCCL_P2P_DISABLE` | `1` | Diagnostic only. If the hang disappears, the problem is peer-to-peer or NVLink. |
| `NCCL_ALGO` | `Ring` | Diagnostic only. Forces a simpler algorithm; useful for isolating tree or NVLS-specific stalls. |

The two `DISABLE` flags are for isolating a fault, not for running in production. If
turning one off fixes your run, you have found where the problem is, not solved it.

## Xid quick reference

| Xid | Name | Usually | RMA candidate | More |
| --- | --- | --- | --- | --- |
| 13 | Graphics Engine Exception | Application bug | No | [details](https://denpex.com/failures/xid-13-pytorch-cuda-oob) |
| 31 | GPU memory page fault (MMU fault) | Application bug | No | [details](https://denpex.com/failures/xid-31-page-fault-memory-pressure) |
| 32 | Invalid or corrupted push buffer stream | Ambiguous | No | [reference](https://denpex.com/xid) |
| 38 | Driver firmware error | Driver or firmware | No | [reference](https://denpex.com/xid) |
| 43 | GPU stopped processing | Application bug | No | [details](https://denpex.com/failures/xid-43-pcie-link-training) |
| 45 | Preemptive cleanup, due to previous errors | Application bug | No | [reference](https://denpex.com/xid) |
| 48 | Double Bit ECC Error (DBE) | Hardware | Yes, on repeat | [details](https://denpex.com/failures/xid-48-dbe-memory-degradation) |
| 61 | Internal micro-controller breakpoint/warning | Driver or firmware | Yes, on repeat | [details](https://denpex.com/failures/nvidia-xid-61-microcontroller-warning) |
| 62 | Internal micro-controller halt | Driver or firmware | Yes, on repeat | [reference](https://denpex.com/xid) |
| 63 | ECC page retirement or row remapping recording event | Hardware | No | [details](https://denpex.com/failures/hardware-ecc-xid-63-storm) |
| 64 | ECC page retirement or row remapper recording failure | Hardware | Yes, on repeat | [details](https://denpex.com/failures/hardware-ecc-xid-64-inforom) |
| 65 | Video processor exception | Ambiguous | No | [reference](https://denpex.com/xid) |
| 69 | Graphics Engine class error | Application bug | No | [details](https://denpex.com/failures/nvidia-xid-69-graphics-engine-class-error) |
| 74 | NVLink error | Hardware | Yes, on repeat | [details](https://denpex.com/failures/nvidia-xid-74-nvlink-fatal-error) |
| 79 | GPU has fallen off the bus | Hardware | Yes, on repeat | [details](https://denpex.com/failures/gpu-fallen-off-bus-xid79) |
| 92 | High single-bit ECC error rate | Hardware | No | [reference](https://denpex.com/xid) |
| 94 | Contained ECC error | Hardware | Yes, on repeat | [reference](https://denpex.com/xid) |
| 95 | Uncontained ECC error | Hardware | Yes, on repeat | [reference](https://denpex.com/xid) |
| 109 | Context switch timeout error | Ambiguous | No | [reference](https://denpex.com/xid) |
| 119 | GSP RPC timeout | Driver or firmware | No | [reference](https://denpex.com/xid) |
| 120 | GSP error | Driver or firmware | No | [reference](https://denpex.com/xid) |
| 140 | ECC unrecovered error | Hardware | Yes, on repeat | [reference](https://denpex.com/xid) |
| 143 | GPU initialization failure | Hardware | Yes, on repeat | [reference](https://denpex.com/xid) |

## Xid detail

### Xid 13: Graphics Engine Exception

**Usually:** Application bug. **RMA candidate:** no, not on its own.

A kernel executed an illegal operation, most often an out-of-bounds memory access or a bad pointer in application or library code. The GPU raised an exception and killed the offending context.

**First action.** Run the workload under compute-sanitizer. Xid 13 is an application bug far more often than a hardware fault, and the sanitizer names the kernel and the access.

Full entry: https://denpex.com/failures/xid-13-pytorch-cuda-oob

### Xid 31: GPU memory page fault (MMU fault)

**Usually:** Application bug. **RMA candidate:** no, not on its own.

The GPU MMU rejected a memory access: the address was unmapped, or the access type was not permitted for that mapping. Usually a bad pointer, a use-after-free, or a race on a freed allocation.

**First action.** compute-sanitizer --tool memcheck. The fault message names the engine and access type, which narrows it to a kernel quickly.

Full entry: https://denpex.com/failures/xid-31-page-fault-memory-pressure

### Xid 32: Invalid or corrupted push buffer stream

**Usually:** Ambiguous. **RMA candidate:** no, not on its own.

The command stream the driver submitted was rejected as malformed. Can be a driver/user-mode bug, and can also be memory corruption on the path that built the buffer.

**First action.** Check for concurrent Xid 48/94/95 (ECC) on the same GPU, corruption upstream explains a malformed stream. Absent that, treat it as a driver/library bug and check versions.

### Xid 38: Driver firmware error

**Usually:** Driver or firmware. **RMA candidate:** no, not on its own.

The driver firmware hit an internal error. Not caused by application code.

**First action.** Compare driver and VBIOS against the fleet baseline; a mismatched pair is the usual cause. Reset the GPU, then update if it recurs.

### Xid 43: GPU stopped processing

**Usually:** Application bug. **RMA candidate:** no, not on its own.

A channel was stopped after an error in the work it was executing. The GPU itself is healthy; the context that faulted was terminated.

**First action.** Read the Xid immediately BEFORE this one, 43 is usually the cleanup that follows another fault, not the fault itself.

Full entry: https://denpex.com/failures/xid-43-pcie-link-training

### Xid 45: Preemptive cleanup, due to previous errors

**Usually:** Application bug. **RMA candidate:** no, not on its own.

The driver tore down a channel because something else already failed. This is TEARDOWN, not a root cause, it is also what appears when a user kills a job with Ctrl-C.

**First action.** Do not diagnose Xid 45. Run `dmesg -T | grep -i xid` and read the Xid immediately BEFORE it; that one is the incident.

### Xid 48: Double Bit ECC Error (DBE)

**Usually:** Hardware. **RMA candidate:** yes, if it repeats on the same serial.

An uncorrectable memory error. ECC caught two flipped bits, which it can detect but not repair, so the data is lost and every context using that memory is suspect.

**First action.** Drain the GPU. Resume from a checkpoint written BEFORE the Xid timestamp, anything computed after it may be silently wrong.

Full entry: https://denpex.com/failures/xid-48-dbe-memory-degradation

### Xid 61: Internal micro-controller breakpoint/warning

**Usually:** Driver or firmware. **RMA candidate:** yes, if it repeats on the same serial.

The GPU management processor (PMU/falcon) hit an internal condition. The GPU is typically unusable until reset, and a repeat on one serial points at silicon or firmware.

**First action.** Reset the GPU (nvidia-smi --gpu-reset -i <id>) or reboot. Check the driver/VBIOS pairing against the fleet baseline.

Full entry: https://denpex.com/failures/nvidia-xid-61-microcontroller-warning

### Xid 62: Internal micro-controller halt

**Usually:** Driver or firmware. **RMA candidate:** yes, if it repeats on the same serial.

The management processor stopped rather than merely warning. Unrecoverable without a reset, and frequently persistent.

**First action.** Drain and reset. A second Xid 62 on the same serial within 30 days is an RMA with the crash dump attached.

### Xid 63: ECC page retirement or row remapping recording event

**Usually:** Hardware. **RMA candidate:** no, not on its own.

The GPU retired a bad memory page (or remapped a row on A100 and later) and recorded it. This is ECC working as designed, a single event is not a failure.

**First action.** Track the RATE, not the event. Check remap pool headroom: nvidia-smi -q -d ROW_REMAPPER. Retirements accelerating over days predict Xid 64.

Full entry: https://denpex.com/failures/hardware-ecc-xid-63-storm

### Xid 64: ECC page retirement or row remapper recording failure

**Usually:** Hardware. **RMA candidate:** yes, if it repeats on the same serial.

The GPU tried to retire a failing page or row and could not record it. The protective mechanism itself has failed, so further errors cannot be contained.

**First action.** RMA. Unlike Xid 63 this is terminal, the spare-row reserve or its bookkeeping is exhausted.

Full entry: https://denpex.com/failures/hardware-ecc-xid-64-inforom

### Xid 65: Video processor exception

**Usually:** Ambiguous. **RMA candidate:** no, not on its own.

The video engine raised an exception, usually during encode or decode work.

**First action.** Confirm a codec workload was running: `nvidia-smi --query-gpu=utilization.encoder,utilization.decoder --format=csv`. Isolate by disabling the hardware codec path.

### Xid 69: Graphics Engine class error

**Usually:** Application bug. **RMA candidate:** no, not on its own.

An invalid class or method was submitted to the graphics engine, a driver or application-level programming error.

**First action.** Run `nvidia-smi` and compare the driver/CUDA pair against the application build. This is not a hardware signal, do not drain the node for it.

Full entry: https://denpex.com/failures/nvidia-xid-69-graphics-engine-class-error

### Xid 74: NVLink error

**Usually:** Hardware. **RMA candidate:** yes, if it repeats on the same serial.

An error on an NVLink connection between GPUs, or between a GPU and an NVSwitch. Collectives crossing the affected link either fail or fall back to a much slower path.

**First action.** nvidia-smi nvlink -e for CRC/replay/recovery counters on BOTH endpoints. A counter ramp usually precedes the Xid by hours.

Full entry: https://denpex.com/failures/nvidia-xid-74-nvlink-fatal-error

### Xid 79: GPU has fallen off the bus

**Usually:** Hardware. **RMA candidate:** yes, if it repeats on the same serial.

The driver can no longer reach the GPU over PCIe. The device stopped answering, power, thermal, link, or board fault. A warm reboot will not bring it back.

**First action.** Capture dmesg, BMC event log and nvidia-bug-report.sh BEFORE power-cycling, the evidence does not survive. Then full chassis power cycle.

Full entry: https://denpex.com/failures/gpu-fallen-off-bus-xid79

### Xid 92: High single-bit ECC error rate

**Usually:** Hardware. **RMA candidate:** no, not on its own.

Correctable errors are being corrected, but at an elevated rate. Nothing has failed yet; this is the early-warning signal that precedes uncorrectable errors.

**First action.** Schedule a drain rather than reacting immediately. Alert on the trend, a rising SBE rate is the cheapest predictor of an upcoming Xid 48/94/95.

### Xid 94: Contained ECC error

**Usually:** Hardware. **RMA candidate:** yes, if it repeats on the same serial.

An uncorrectable error that the GPU CONTAINED to the offending context. Other work on the GPU is unaffected; the faulting process is killed.

**First action.** Kill and restart the affected job. Check row-remap headroom, containment relies on spare rows being available.

### Xid 95: Uncontained ECC error

**Usually:** Hardware. **RMA candidate:** yes, if it repeats on the same serial.

An uncorrectable error the GPU could NOT contain. Corruption reached wider GPU state, so all work on the device is suspect, not just the faulting process.

**First action.** Drain the node. Resume from a checkpoint predating the Xid; do not trust the most recent optimizer state.

### Xid 109: Context switch timeout error

**Usually:** Ambiguous. **RMA candidate:** no, not on its own.

The GPU did not complete a context switch within its timeout. Can be an extremely long-running kernel, and can be a driver or hardware fault.

**First action.** Check for a kernel with an unbounded loop first. If the workload is normal, treat it as a driver-level fault and check the version.

### Xid 119: GSP RPC timeout

**Usually:** Driver or firmware. **RMA candidate:** no, not on its own.

The driver waited for a reply from the GPU System Processor firmware and did not get one. A firmware-side stall rather than an application fault.

**First action.** Reset the GPU. GSP firmware issues are strongly driver-version dependent, check NVIDIA release notes for your exact version before RMA.

### Xid 120: GSP error

**Usually:** Driver or firmware. **RMA candidate:** no, not on its own.

The GPU System Processor reported an internal error. GSP runs much of the driver logic on the GPU itself from R515 onward, so a GSP fault takes the device out even though no application did anything wrong.

**First action.** Reset with `nvidia-smi --gpu-reset -i <id>`, then match the driver against NVIDIA release notes. GSP can be disabled via the NVreg_EnableGpuFirmware kernel module parameter on known-affected versions.

### Xid 140: ECC unrecovered error

**Usually:** Hardware. **RMA candidate:** yes, if it repeats on the same serial.

An ECC error the GPU could not recover from. Treat as uncontained corruption.

**First action.** Drain, resume from a checkpoint predating the event, and collect `nvidia-smi -q -d ECC,ROW_REMAPPER` output for the RMA.

### Xid 143: GPU initialization failure

**Usually:** Hardware. **RMA candidate:** yes, if it repeats on the same serial.

The GPU failed to initialize. It will not be usable until reset, and often not after.

**First action.** Full power cycle, not a warm reboot. Then `nvidia-smi -L` to confirm enumeration. If it recurs on the same slot, test whether the fault follows the card.

## Where each tool stops

Debugging a GPU cluster takes several layers, and knowing which one answers your
question saves more time than any single tool.

| Layer | Answers | Stops at |
| --- | --- | --- |
| NVIDIA DCGM | Is this GPU healthy: Xid, ECC, thermals, throttle reasons | The device boundary. No model of your rank topology. |
| Prometheus and Grafana | What happened across the fleet over time | Numbers. A crash's key artifact is a string. |
| Slurm | What ran, where, and how it terminated | The task boundary. Exit code 1 is not a diagnosis. |
| Ray | Keeping the workload running; actor supervision | Supervision. A RayActorError says nothing about the CUDA event that caused it. |
| Weights & Biases | How this run compares to your other runs | The experiment. It shows the NaN, not which rank produced it first. |
| PyTorch flight recorder, `NCCL_DEBUG` | What the collectives were doing | Raw evidence. Both must be enabled before the failure. |

Longer version, with citations to each project's own docs:
https://denpex.com/solutions/best-tools-gpu-cluster-debugging

## Full encyclopedia

471 failure modes with a verbatim error signature, a root cause, a fix
and citations to upstream vendor documentation or issue trackers:

https://denpex.com/failures

To diagnose your own log rather than browse, paste it into https://denpex.com/console. No account
and no card; it is the free tier of the tool this data comes from.

## Contributing

Corrections are very welcome, especially to the Xid table. If a code is misclassified
or a first action is wrong in your environment, open an issue with the log line and
what actually resolved it.

Signature and Xid data are generated, so send corrections as an issue rather than a
pull request against `signatures.json`, which would be overwritten on the next
regeneration.

## Licence

Data and documentation: CC BY 4.0. Code: MIT. Attribution to Denpex (https://denpex.com) is
appreciated but the licence terms are what govern.
