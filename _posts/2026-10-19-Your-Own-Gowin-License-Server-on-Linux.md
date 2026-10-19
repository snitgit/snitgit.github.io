---
layout: post
title:  "Your Own Gowin License Server on Linux: Keeping an Automated FPGA Flow Alive"
date:   2026-10-19 16:00:00 +0700
categories: FPGA EDA
tags: [gowin, fpga, license-server, tang-mega-138k, linux, systemd, eda, ai-agents, thailand]
---

*[ฉบับภาษาไทย](/posts/Your-Own-Gowin-License-Server-on-Linux-TH/)*

An automated hardware-design flow is only as reliable as its least reliable dependency. For us, that dependency turned out to be a license. This post shows how to run your own Gowin floating license server on Linux. It takes about an hour once you know the three things the documentation doesn't tell you.

## Why it mattered

Our group runs an agentic flow in which an LLM writes Verilog and a pipeline of deterministic gates checks it: lint, simulation bit-exact against a reference model, then synthesis and place-and-route on a Gowin GW5AST-138 (the chip on the Sipeed Tang Mega 138K). The method is described in our APSIPA ASC 2026 paper [1]. A flow like this runs unattended for hours. If the synthesis tool can't get a license at 2 a.m., the run doesn't fail loudly. It just stops producing results.

Like many Tang board users, we had been using a shared online license server. It is a generous free service and it got us started. But one day `gw_sh` printed:

```
The license expires in 5 days.
```

The countdown belonged to the shared server, not to us. A research cluster needs something it controls.

## What you need

| Item | Where it comes from |
|---|---|
| A **floating** license file (`gowin_E_<MAC>_floating.lic`) | Gowin's "Apply License" form, free for academic use |
| **Gowin License Server for Linux** | Gowin's documentation database (member login required) |
| A Linux machine with a **stable MAC address** | the license is locked to it |
| Gowin EDA **V1.9.8.01 or later** on the clients | we tested V1.9.12.02 |

### 1. Apply for the license

On the Gowin website, open Support → **Apply License** and choose:

- **Type of License:** Shared (Floating for license Server)
- **Operating System:** Linux
- **PC MAC Address:** the MAC of the machine that will *run the server* (`cat /sys/class/net/<nic>/address`)

Our reply came within a day: a 10-seat floating license, valid for one year. Check it before going further:

```bash
grep -E "MODE|HOST_ID|EXP_DATE|COUNT" gowin_E_*_floating.lic
# MODE = FLOATING
# HOST_ID = <your MAC, no colons, upper case>
```

### 2. Pick the right server package

Search the documentation database for "license server for linux". You will find three packages, and the names are confusing:

| Listing | Use it? |
|---|---|
| For Gowin_V1.9.8 and earlier version (2022, ZIP) | No: old license format |
| **For Gowin_V1.9.8.01 and Higher version** (30/12/2024, GZ) | **Yes**: matches the `gowin_E_*.lic` format; works with V1.9.12.02 |
| For Gowin_V1.9.12.01 and earlier version (2.2E, 31/12/2025, GZ) | Backup, if the one above ever rejects a newer EDA |

The package (`Gowin_LicenseServer_V2.1_linux.tar.gz`) contains one program, `bin/license_server`, plus libraries. There is no README.

### 3. Install, and avoid the crash

```bash
mkdir -p /opt/gowin-license && cd /opt/gowin-license
tar xzf Gowin_LicenseServer_V2.1_linux.tar.gz
cp gowin_E_*_floating.lic . && chmod 600 gowin_E_*_floating.lic
```

**Gotcha #1.** If you put the package's own `lib/` folder on `LD_LIBRARY_PATH`, the server dies at once:

```
symbol lookup error: .../lib/libpthread.so.0: undefined symbol: __libc_dl_error_tsd, version GLIBC_PRIVATE
```

The package ships its own copy of `libpthread`, which clashes with a modern glibc. Use **only** `bin/`, which holds the Qt libraries the server actually needs:

```bash
S=/opt/gowin-license/Gowin_LicenseServer_V2.1_linux
QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH=$S/bin $S/bin/license_server --help
#   -s, --license <file>  Specify the license file. Default is 'gowin.lic'
#   -p, --port <port>     Specify listening port number. Default is 10559
#   -o, --out <location>  Specify the log file location. Default is '.'
```

### 4. Run it as a service

```ini
# /etc/systemd/system/gowin-license.service
[Unit]
Description=Gowin floating license server
After=network-online.target

[Service]
User=gowin
Environment=QT_QPA_PLATFORM=offscreen
Environment=LD_LIBRARY_PATH=/opt/gowin-license/Gowin_LicenseServer_V2.1_linux/bin
ExecStart=/opt/gowin-license/Gowin_LicenseServer_V2.1_linux/bin/license_server \
    -s /opt/gowin-license/gowin_E_XXXXXXXXXXXX_floating.lic -p 10559 -o /opt/gowin-license/log
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now gowin-license
```

The log confirms it:

```
Server started.
Listening Port: 10559
License Count: 10
```

### 5. Point the clients at it

**Gotcha #2.** `gw_sh` reads the server address only from `bin/gwlicense.ini` inside each Gowin installation. Setting `LM_LICENSE_FILE`, as many build scripts do, has no effect.

```ini
[license]
lic="localhost:10559"
```

From other machines, use the server's hostname instead of `localhost`. In the IDE, Help → Manage License does the same.

**Gotcha #3.** Pointing `lic=` at the `.lic` file itself does **not** work for a floating license:

```
License verification failed  License mode not match.
```

A floating license must be served; only the server reads the file.

### 6. Test it properly

`gw_sh --version` should print its banner **without** a "license expires" line, and the server log should show a check-out:

```
Checkin
user: alice@buildhost
total: 10, used: 0, remain: 10
```

A passing version check doesn't prove device entitlement, though. Build something small for your actual part:

```tcl
set_device -name GW5AST-138C GW5AST-LV138PG484AC1/I0
add_file cnt.v
set_option -top_module cnt
run all
```

If `impl/pnr/cnt.fs` appears, the whole path works: license, device, synthesis, place-and-route and bitstream.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `symbol lookup error ... libpthread.so.0` | `LD_LIBRARY_PATH` includes the package `lib/`; use `bin/` only |
| `License mode not match` | the client points at the `.lic` file; point it at `host:port` |
| `The license expires in N days` | the client still uses another server; check `gwlicense.ini` |
| Server refuses the license | the MAC changed; compare `HOST_ID` with `/sys/class/net/<nic>/address` |
| Clients can't connect | is the port listening (`ss -ltn \| grep 10559`)? Is a firewall in the way? |

## Lessons

1. **Don't put it in a container.** The license is locked to a MAC address. A Kubernetes pod gets a new random MAC every time it restarts, so the license fails. Run the server on a fixed machine and let containers connect to it over the network.
2. **Virtual machines need a pinned MAC.** Our server runs on a KVM VM. Its MAC stays the same unless someone recreates the network card, so we wrote it into the admin notes.
3. **The server has no authentication.** Anyone who can reach the port can take a seat. Firewall it to your own network.
4. **Seats are shared.** We opened ours to the whole campus. That's good for students, but a full server also blocks your own overnight runs, so ask for enough seats.
5. **Put the renewal in your calendar.** Floating licenses expire. Re-apply a month early.

## Why this is worth doing for Thai universities

Gowin's devices and boards are inexpensive, and the license for academic use is free. That makes them a realistic platform for teaching FPGA design and for research that needs real silicon, not just simulation. What's missing is the operational know-how: one afternoon of setup that each lab tends to rediscover alone. A faculty that runs its own license server can share it across campus. It owns that piece of its toolchain instead of borrowing it.

## References

[1] S. Sanghlao and S. Sanglao, "Agentic LLM-Driven RTL Design for Factor-Graph Processing Elements Using a Containerized EDA Loop," *Proc. APSIPA ASC 2026*, to appear.
