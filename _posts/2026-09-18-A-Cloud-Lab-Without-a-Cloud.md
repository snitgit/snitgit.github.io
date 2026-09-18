---
layout: post
title:  "A Cloud Lab Without a Cloud: Virtual Network Labs on a University Slurm Cluster"
date:   2026-09-18 16:00:00 +0700
categories: HPC Education
tags: [slurm, virtual-lab, qemu, kvm, private-cloud, labs-on-demand, ai-agents, thailand]
---

A network-management course needs dozens of machines that students can break, fix and break again. Most universities rent them from a public cloud. This post is about doing it on the HPC cluster the faculty already owns. It covers what worked, what cost us a day, and why I think "cluster infrastructure as a class" is a better default for Thai universities than a public-cloud account.

## The problem

The course has 37 students in 13 groups. A typical lab gives each group four machines, for example a workstation, an SSH jump host and two targets. Later labs add routers, an SNMP manager, a NetFlow collector, a Zabbix server and an Ansible controller. That is 52 or more machines per session, three hours a week, for a whole semester:

> 52 VMs × 3 hours × ~10 lab weeks ≈ **1,500+ VM-hours per term**, before anyone practises outside class.

A public cloud handles this easily, and that is exactly the trap. The bill is in dollars, it arrives after the fact, it needs a credit card or a procurement process that was designed for buying furniture, and it rises with every student who forgets to shut a VM down. Meanwhile the faculty's own cluster is 2 × 256 cores and about a terabyte of RAM. Between research jobs it sits idle, already paid for.

The obvious question: can an HPC cluster behave like a small private cloud for one afternoon a week?

## HPC is not a cloud (and that's fine)

A cluster scheduler is built to run batch jobs for users who are *not* trusted with the machine. Everything a cloud does to give you a VM, an HPC site deliberately takes away:

| A cloud expects… | Our cluster actually gives | What we did instead |
|---|---|---|
| Root on the host | A normal user, **zero Linux capabilities** | Unprivileged QEMU/KVM inside a Singularity container |
| Bridges, VLANs, virtual switches | `ip link add` → *Operation not permitted* | QEMU user-mode networking: every VM gets its own private NAT |
| Long-lived VMs | Every process is killed when the job ends | One Slurm job = one class session; VMs live exactly as long as it does |
| Public IPs / security groups | A campus firewall and one reachable node | Fixed port per VM: `8000 + group×1000 + node×100 + 22` |
| Block storage volumes | Fast local NVMe (wiped on reboot) + a shared parallel filesystem | Copy-on-write disks on NVMe, backed up to the parallel filesystem after each session |

None of these needed an administrator. That matters: the fastest way to kill an idea like this is to make it depend on someone else's change window.

## The architecture in one picture

```
  student laptop                     compute node (one Slurm job, 3h)
  ──────────────                     ─────────────────────────────────────────────
                                      qemu  g5-pc        ── NAT ── :13022
  ssh -p 13122 ───────────────────▶   qemu  g5-jumphost  ── NAT ── :13122
                                      qemu  g5-target-a  ── NAT ── :13222
                                      qemu  g5-target-b  ── NAT ── :13322
                                        │   (×13 groups = 52 VMs)
                                        │
          base image (read-only) ◀──────┤  each disk = tiny copy-on-write overlay
                                        │
   parallel filesystem ◀── backup ──────┘  at session end: power off, copy overlays
   (survives node reboots)                 next session: restore, students resume
```

The controller is about 250 lines of Python. It reads a JSON topology per group, starts one QEMU process per node, and gives each VM its identity (hostname, instance ID) through the SMBIOS serial string that cloud-init reads. It shuts VMs down through ACPI so guests flush their disks before the job ends. There is no daemon, database or web server.

What it achieved on the first full-scale run:

- **52 VMs booted, 0 failures**; SSH answering on all 52 within **35 seconds**
- A real login and a jump-host hop verified automatically for **every group**
- **57 GB of 503 GB RAM** used on one node, so the heavier labs have room
- Clean power-off of every VM in **8 seconds**; about **100 MB** of state per VM saved between sessions
- A simulated node reboot: disks and base image restored from the shared filesystem, student files intact, **SSH host keys unchanged**

## Lessons learned (the honest part)

**1. `--time=04:00` is four *minutes*.** Slurm reads `MM:SS` before `HH:MM:SS`. An earlier iteration of this project wrote `--time=00:15` and `--time=04:00`, saw jobs die at exactly one minute, and concluded that the cluster had a hard 60-second limit. A whole layer of workarounds followed: chunked downloads, resumable copies, jobs split into sub-minute steps. One `sacct` query showed every job's limit had been 00:01:00, and the partition's real limit was *unlimited*. The lesson is older than HPC: **measure the constraint before you engineer around it.**

**2. The capability wall is only on the host.** Lab platforms such as PNETLab, EVE-NG, GNS3 and ContainerLab need to create bridges and virtual cables, and we could not do that on the host. Inside a VM, though, we are root on our own kernel. The cluster's CPUs support nested virtualization, so a guest can run its own hypervisor. This moves "root" down one layer instead of asking anyone for it. The pilot worked on the first try: PNETLab installed from its official repository inside an Ubuntu VM, booted its own kernel, built the ten bridges our account may not create on the host, and served its web UI through a forwarded port. The whole build took 25 minutes and needed no administrator.

**3. A router image belongs to its hardware.** The original plan was to copy the IOS image off a physical Cisco 1841 and run it in an emulator for every group. It doesn't work: an IOS image only boots on its own platform, and that emulator doesn't emulate the 1841. It is also outside the image's licence. An open network OS (VyOS) runs unmodified in the same VMs, installs unattended in six minutes, and costs nothing per copy.

**4. Mirrors matter more than cores.** Through the campus proxy, the official Ubuntu archive delivered about **0.25 MB/s**. Mirrors hosted in Thailand delivered **3.5–8 MB/s**, roughly 15–30× faster. On a platform that builds images, the choice of mirror decided whether a step took an hour or three minutes.

**5. Two orchestrators on one node is one too many.** At one point two independent sessions submitted jobs to the same node. Each one started by "cleaning up" stray VMs, and each killed the other's machines. Shared state needs one owner, or per-tenant namespaces. This matters more now that the sessions in question are increasingly AI agents rather than people (see below).

## The proposal: cluster infrastructure *as a class*

The pieces a public cloud sells already exist on most university clusters. They are just named differently:

| Cloud concept | Already on the cluster |
|---|---|
| Placement / scheduler | Slurm |
| Hypervisor | KVM via unprivileged QEMU |
| Machine images | Read-only base images + copy-on-write overlays |
| Durable volumes | The parallel filesystem (Lustre, GPFS…) |
| Instance metadata | cloud-init (NoCloud seed + SMBIOS) |
| Billing | Fair-share accounting that already exists |

What's missing is the **on-demand** part. Today one job holds the whole class for the whole afternoon. The next step is **labs on demand**:

1. **One job per group, started when asked.** A student clicks "start my lab"; a job starts that group's topology, and the page shows their ports.
2. **Idle reaping and quotas** through Slurm's own limits, so a forgotten lab does not hold a node overnight.
3. **Reset and snapshot** by discarding or keeping an overlay: a broken lab is a one-second fix.
4. **Platform-in-a-VM when needed:** PNETLab or GNS3 inside a VM for courses that need a topology GUI and real switching, and plain VMs for courses that don't.

This is not a public cloud and should not try to be one. It is a teaching cloud: it runs on hardware the university already owns, uses scheduling policy it already has, and keeps student data inside the university, which is a real consideration under Thailand's PDPA. The same pattern extends beyond networking. A programming, app-development or data-science class is just a different base image.

## Where AI agents come in

There is a second reason to build this now. AI agents need exactly the same thing students do: **a disposable, isolated machine they can break.** An agent that configures routers, runs Ansible or debugs a service should do it in a sandbox, not on production. A lab-on-demand service provides that primitive: a topology of fresh VMs, reachable over SSH, gone when the job ends.

That opens several uses beyond the classroom:

- **Rehearse the lab.** Before class, an agent works through each lab sheet as a student would, on a fresh topology. If the instructions are wrong, the agent finds out before 37 students do.
- **Grade by inspection.** After class, an agent logs into each group's VMs and checks the actual configuration, not a screenshot of it.
- **Research at ensemble scale.** Evaluating network-operations agents means many independent trials. In the terms of [an earlier post]({% post_url 2026-07-09-The-Cluster-and-the-Bound %}), a swarm of agents, each with its own lab, is a *level-A* workload, the kind of parallelism a cluster is best at.

Lesson 5 applies here twice over. The first time two agents shared our node, they destroyed each other's work. Per-tenant isolation is a prerequisite for running agents at all.

## Why this is a good bet for Thai universities

As with formal verification, the asymmetry is the argument. Renting cloud capacity is **budget-bound**: the cost scales with every student and every semester, and it leaves the university with nothing when the credits run out. Building a teaching cloud on existing clusters is **skill-bound**. It needs a few people who understand schedulers, hypervisors and images, and what they build stays with the institution. Many Thai universities already run clusters, and the national research network links them. Base images, topologies and lab sheets could be shared across institutions as easily as papers.

Three first steps, in order of effort:

1. **Check your cluster honestly.** Is `/dev/kvm` available inside a job? Is nested virtualization on? What is the real time limit (ask `sacct`, not memory)? Which mirror is fastest from inside a job?
2. **Run one class on it.** One job, one topology per group, overlays on local disk, backups on the shared filesystem. It takes less code than you would expect.
3. **Make it on-demand, then share it.** Per-group jobs and a small portal turn a class session into a service, and agents can use that service just as students do.

The cloud we need for teaching is not somewhere else. It is in the machine room, idle every afternoon.
