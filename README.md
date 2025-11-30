# Evaluating Log Probs Obtained from SGLang v/s HuggingFace Implementations for a MCQ Task

I noticed significant discrepancies and this is my attempt to create a clean reproducible testbed for anyone to see these differences.

Note: Before anyone points out that SGLang does not return log probabilities but normalized log probabilities, don't worry I've already taken that into account in my analysis by using the `Token Length Normalized` choices method which can be easily denormalized by scaling with number of tokens in the option.

# HF Run - 

## Env - 

I used uv for my environment and to share my exact setup I ran `uv pip list > requirements.txt` and the output is present in `requirements.txt`

## Run Command -

CUDA_VISIBLE_DEVICES=0 python get_hf_log_probs.py

# SGL Run - 

## Env -

Exact Image as sglang:latest points to different images over time - https://hub.docker.com/layers/lmsysorg/sglang/latest/images/sha256-3bcc7b2db268fe8d965090f91b0dcd6663c5571b690df39ebb6131abad2a2ad8

Change mounts in the command below as necessary -

```
docker run --gpus all -it \
    --shm-size 32g \
    -v "Reproducible_Example_SGL_vs_HF":"Reproducible_Example_SGL_vs_HF" \
    -v "/NS/llm-artifacts/nobackup/HF_HOME":"/NS/llm-artifacts/nobackup/HF_HOME" \
    --ipc=host \
    lmsysorg/sglang:latest \
    bash
```


## Run Command -

bash run_sgl_configs.sh

Ran multiple attention backends with and without determinism

# Machine - 

## OS / Kernel -

Commmand - `uname -a`

Output - `Linux sws-2h100-05 6.6.88.1.amd64-smp #1 SMP PREEMPT_DYNAMIC Mon Apr 28 14:29:46 CEST 2025 x86_64 GNU/Linux`

Command - `lsb_release -a`

Output - 
```
No LSB modules are available.
Distributor ID: Debian
Description:    Debian GNU/Linux 12 (bookworm)
Release:        12
Codename:       bookworm
```

## CPU Info -

Command - `lscpu`

Output - 
```
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 57 bits virtual
  Byte Order:             Little Endian
CPU(s):                   48
  On-line CPU(s) list:    0-47
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Xeon(R) Gold 5317 CPU @ 3.00GHz
    CPU family:           6
    Model:                106
    Thread(s) per core:   2
    Core(s) per socket:   12
    Socket(s):            2
    Stepping:             6
    CPU(s) scaling MHz:   25%
    CPU max MHz:          3600.0000
    CPU min MHz:          800.0000
    BogoMIPS:             6000.00
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs 
                          bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt ts
                          c_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fs
                          gsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xget
                          bv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req vnmi avx512vbmi umip pku ospke av
                          x512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
Virtualization features:  
  Virtualization:         VT-x
Caches (sum of all):      
  L1d:                    1.1 MiB (24 instances)
  L1i:                    768 KiB (24 instances)
  L2:                     30 MiB (24 instances)
  L3:                     36 MiB (2 instances)
NUMA:                     
  NUMA node(s):           2
  NUMA node0 CPU(s):      0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46
  NUMA node1 CPU(s):      1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47
Vulnerabilities:          
  Gather data sampling:   Mitigation; Microcode
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Mitigation; Clear CPU buffers; SMT vulnerable
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Not affected
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI SW loop, KVM SW loop
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

## Memory -

Command - `free -h`

Output - 
```
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti       240Gi        13Gi        12Mi       760Gi       766Gi
Swap:          2.0Gi       484Mi       1.5Gi
```

## GPU Information -

Command - `nvidia-smi -L`

Output - 
```
GPU 0: NVIDIA H100 NVL (UUID: <REMOVED>)
GPU 1: NVIDIA H100 NVL (UUID: <REMOVED>)
```

Command - `nvidia-smi`

Output -
```
Sun Nov 30 14:15:01 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 NVL                On  |   00000000:17:00.0 Off |                    0 |
| N/A   36C    P0             83W /  400W |     529MiB /  95830MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 NVL                On  |   00000000:CA:00.0 Off |                    0 |
| N/A   32C    P0             60W /  400W |       3MiB /  95830MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A   2201882      C   ...niconda3/envs/lke_stable/bin/python        516MiB |
+-----------------------------------------------------------------------------------------+
```

Command - `cat /proc/driver/nvidia/version`

Output -
```
NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.14  Thu Feb 22 01:44:30 UTC 2024
GCC version:  gcc version 12.2.0 (Debian 12.2.0-14+deb12u1) 
```

Command - `nvcc --version`

Output - 
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```