---
layout: post
title:  "ตั้งเซิร์ฟเวอร์ License ของ Gowin เองบน Linux: ให้ flow ออกแบบ FPGA อัตโนมัติทำงานได้ไม่สะดุด"
date:   2026-10-19 16:30:00 +0700
categories: FPGA EDA
tags: [gowin, fpga, license-server, tang-mega-138k, linux, systemd, eda, ai-agents, thailand]
---

*[English version](/posts/Your-Own-Gowin-License-Server-on-Linux/)*

flow ออกแบบฮาร์ดแวร์อัตโนมัติจะเชื่อถือได้แค่ไหน ขึ้นกับส่วนที่เชื่อถือได้น้อยที่สุดของมัน สำหรับเรา ส่วนนั้นกลับกลายเป็น license บทความนี้อธิบายวิธีตั้งเซิร์ฟเวอร์ floating license ของ Gowin เองบน Linux ใช้เวลาประมาณหนึ่งชั่วโมง ถ้ารู้สามเรื่องที่เอกสารไม่ได้บอกไว้

## ทำไมเรื่องนี้สำคัญ

กลุ่มวิจัยของเราใช้ agentic flow ที่ให้ LLM เขียน Verilog แล้วส่งผ่านด่านตรวจแบบ deterministic หลายด่าน ได้แก่ lint, simulation ที่ต้องตรงกับ reference model แบบ bit-exact แล้วจึง synthesis และ place-and-route บนชิป Gowin GW5AST-138 (ชิปบนบอร์ด Sipeed Tang Mega 138K) วิธีการนี้อธิบายไว้ในบทความของเราที่ APSIPA ASC 2026 [1] flow แบบนี้รันเองโดยไม่มีคนเฝ้าเป็นชั่วโมง ๆ ถ้าเครื่องมือ synthesis ขอ license ไม่ได้ตอนตีสอง มันจะไม่ล้มให้เห็นชัด ๆ แต่จะหยุดผลิตผลลัพธ์ไปเฉย ๆ

เหมือนผู้ใช้บอร์ด Tang หลายคน เราเคยใช้เซิร์ฟเวอร์ license ออนไลน์ที่ใช้ร่วมกัน ซึ่งเป็นบริการฟรีที่มีน้ำใจมาก และช่วยให้เราเริ่มต้นได้ แต่วันหนึ่ง `gw_sh` ก็แสดงข้อความว่า

```
The license expires in 5 days.
```

ตัวนับถอยหลังนี้เป็นของเซิร์ฟเวอร์ที่ใช้ร่วมกัน ไม่ใช่ของเรา คลัสเตอร์วิจัยต้องมีสิ่งที่ตัวเองควบคุมได้

## สิ่งที่ต้องมี

| รายการ | ได้มาจาก |
|---|---|
| ไฟล์ **floating** license (`gowin_E_<MAC>_floating.lic`) | แบบฟอร์ม "Apply License" ของ Gowin ฟรีสำหรับงานวิชาการ |
| **Gowin License Server for Linux** | ฐานข้อมูลเอกสารของ Gowin (ต้องล็อกอินสมาชิก) |
| เครื่อง Linux ที่มี **MAC address คงที่** | license ผูกกับ MAC นี้ |
| Gowin EDA **V1.9.8.01 ขึ้นไป** บนเครื่องผู้ใช้ | เราทดสอบกับ V1.9.12.02 |

### 1. ขอ license

ที่เว็บไซต์ Gowin เข้าเมนู Support → **Apply License** แล้วเลือก

- **Type of License:** Shared (Floating for license Server)
- **Operating System:** Linux
- **PC MAC Address:** MAC ของเครื่องที่จะ *รันเซิร์ฟเวอร์* (`cat /sys/class/net/<nic>/address`)

เราได้รับคำตอบภายในหนึ่งวัน เป็น floating license 10 seat ใช้ได้หนึ่งปี (ภายหลัง Gowin เพิ่มให้เป็น 50 seat ดูในหัวข้อบทเรียนด้านล่าง) ตรวจไฟล์ก่อนทำขั้นต่อไป

```bash
grep -E "MODE|HOST_ID|EXP_DATE|COUNT" gowin_E_*_floating.lic
# MODE = FLOATING
# HOST_ID = <your MAC, no colons, upper case>
```

### 2. เลือกแพ็กเกจเซิร์ฟเวอร์ให้ถูก

ค้นในฐานข้อมูลเอกสารด้วยคำว่า "license server for linux" จะพบสามแพ็กเกจ และชื่อชวนสับสน

| รายการที่พบ | ใช้ได้ไหม |
|---|---|
| For Gowin_V1.9.8 and earlier version (2022, ZIP) | ไม่ได้: รูปแบบ license เก่า |
| **For Gowin_V1.9.8.01 and Higher version** (30/12/2024, GZ) | **ใช้ตัวนี้**: ตรงกับรูปแบบ `gowin_E_*.lic` และใช้กับ V1.9.12.02 ได้ |
| For Gowin_V1.9.12.01 and earlier version (2.2E, 31/12/2025, GZ) | ตัวสำรอง เผื่อวันหนึ่งตัวข้างบนไม่รับ EDA รุ่นใหม่ |

แพ็กเกจ (`Gowin_LicenseServer_V2.1_linux.tar.gz`) มีโปรแกรมเดียวคือ `bin/license_server` กับไลบรารีต่าง ๆ ไม่มี README

### 3. ติดตั้ง และเลี่ยงปัญหาโปรแกรมล่ม

```bash
mkdir -p /opt/gowin-license && cd /opt/gowin-license
tar xzf Gowin_LicenseServer_V2.1_linux.tar.gz
cp gowin_E_*_floating.lic . && chmod 600 gowin_E_*_floating.lic
```

**กับดักที่ 1** ถ้าใส่โฟลเดอร์ `lib/` ของแพ็กเกจไว้ใน `LD_LIBRARY_PATH` เซิร์ฟเวอร์จะล่มทันที

```
symbol lookup error: .../lib/libpthread.so.0: undefined symbol: __libc_dl_error_tsd, version GLIBC_PRIVATE
```

แพ็กเกจแถม `libpthread` ของตัวเองมาด้วย ซึ่งชนกับ glibc รุ่นใหม่ ให้ใช้ **เฉพาะ** `bin/` ซึ่งมีไลบรารี Qt ที่เซิร์ฟเวอร์ต้องใช้จริง

```bash
S=/opt/gowin-license/Gowin_LicenseServer_V2.1_linux
QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH=$S/bin $S/bin/license_server --help
#   -s, --license <file>  Specify the license file. Default is 'gowin.lic'
#   -p, --port <port>     Specify listening port number. Default is 10559
#   -o, --out <location>  Specify the log file location. Default is '.'
```

### 4. รันเป็น service

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

log ยืนยันว่าเซิร์ฟเวอร์ทำงานแล้ว

```
Server started.
Listening Port: 10559
License Count: 50
```

### 5. ตั้งค่าเครื่องผู้ใช้ให้ชี้มาที่เซิร์ฟเวอร์

**กับดักที่ 2** `gw_sh` อ่านที่อยู่เซิร์ฟเวอร์จาก `bin/gwlicense.ini` ในโฟลเดอร์ติดตั้ง Gowin แต่ละชุดเท่านั้น การตั้ง `LM_LICENSE_FILE` อย่างที่ build script หลายตัวทำ ไม่มีผลใด ๆ

```ini
[license]
lic="localhost:10559"
```

ถ้าเรียกจากเครื่องอื่น ให้ใช้ชื่อเครื่องเซิร์ฟเวอร์แทน `localhost` ใน IDE ใช้เมนู Help → Manage License ได้ผลเหมือนกัน

**กับดักที่ 3** การชี้ `lic=` ไปที่ไฟล์ `.lic` โดยตรง **ใช้ไม่ได้** กับ floating license

```
License verification failed  License mode not match.
```

floating license ต้องให้บริการผ่านเซิร์ฟเวอร์ มีแต่เซิร์ฟเวอร์เท่านั้นที่อ่านไฟล์นี้

### 6. ทดสอบให้ถึงจริง

`gw_sh --version` ควรแสดง banner **โดยไม่มี** บรรทัด "license expires" และ log ของเซิร์ฟเวอร์ควรมีบันทึกการยืม license

```
Checkin
user: alice@buildhost
total: 50, used: 0, remain: 50
```

แต่การตรวจเวอร์ชันผ่านไม่ได้พิสูจน์ว่า license ครอบคลุมชิปของเรา ให้ลอง build วงจรเล็ก ๆ สำหรับชิปที่ใช้จริง

```tcl
set_device -name GW5AST-138C GW5AST-LV138PG484AC1/I0
add_file cnt.v
set_option -top_module cnt
run all
```

ถ้าได้ไฟล์ `impl/pnr/cnt.fs` แปลว่าทั้งเส้นทางใช้ได้ ทั้ง license, ชิป, synthesis, place-and-route และ bitstream

## การแก้ปัญหา

| อาการ | วิธีแก้ |
|---|---|
| `symbol lookup error ... libpthread.so.0` | `LD_LIBRARY_PATH` มี `lib/` ของแพ็กเกจอยู่ ให้ใช้ `bin/` อย่างเดียว |
| `License mode not match` | เครื่องผู้ใช้ชี้ไปที่ไฟล์ `.lic` ให้ชี้ไปที่ `host:port` แทน |
| `The license expires in N days` | เครื่องผู้ใช้ยังใช้เซิร์ฟเวอร์อื่นอยู่ ตรวจ `gwlicense.ini` |
| เซิร์ฟเวอร์ไม่รับ license | MAC เปลี่ยน เทียบ `HOST_ID` กับ `/sys/class/net/<nic>/address` |
| เครื่องผู้ใช้เชื่อมต่อไม่ได้ | พอร์ตเปิดรออยู่ไหม (`ss -ltn \| grep 10559`) มี firewall ขวางอยู่หรือเปล่า |

## บทเรียน

1. **อย่าเอาไปใส่ container** license ผูกกับ MAC address แต่ pod ใน Kubernetes ได้ MAC สุ่มใหม่ทุกครั้งที่ restart license จึงใช้ไม่ได้ ให้รันเซิร์ฟเวอร์บนเครื่องที่คงที่ แล้วให้ container เชื่อมต่อผ่านเครือข่าย
2. **virtual machine ต้องล็อก MAC ไว้** เซิร์ฟเวอร์ของเรารันบน VM แบบ KVM MAC จะไม่เปลี่ยนตราบใดที่ไม่มีใครสร้างการ์ดเครือข่ายใหม่ เราจึงจดไว้ในบันทึกของผู้ดูแลระบบ
3. **เซิร์ฟเวอร์ไม่มีการยืนยันตัวตน** ใครเข้าถึงพอร์ตได้ก็ยืม seat ได้ ให้ตั้ง firewall จำกัดไว้เฉพาะเครือข่ายของเรา
4. **seat ใช้ร่วมกัน** เราเปิดให้ทั้งวิทยาเขตใช้ ซึ่งดีสำหรับนักศึกษา แต่ถ้า seat เต็ม งานที่รันข้ามคืนของเราเองก็ติดด้วย จึงควรขอ seat ให้พอ เมื่อเราขอเพิ่ม Gowin ออก license ใหม่ให้เป็น 50 seat แทน 10 seat ภายในวันเดียวกัน
5. **ลงวันต่ออายุไว้ในปฏิทิน** floating license มีวันหมดอายุ ให้ขอใหม่ล่วงหน้าหนึ่งเดือน

## ทำไมเรื่องนี้คุ้มค่าสำหรับมหาวิทยาลัยไทย

ชิปและบอร์ดของ Gowin ราคาไม่แพง และ license สำหรับงานวิชาการก็ฟรี จึงเป็นแพลตฟอร์มที่ใช้ได้จริงทั้งสำหรับสอนออกแบบ FPGA และงานวิจัยที่ต้องการซิลิคอนจริง ไม่ใช่แค่ simulation สิ่งที่ขาดคือความรู้เชิงปฏิบัติการ เป็นงานตั้งค่าครึ่งวันที่แต่ละแล็บมักต้องค้นพบเองซ้ำ ๆ คณะที่รันเซิร์ฟเวอร์ license ของตัวเองแบ่งให้ทั้งวิทยาเขตใช้ได้ และเป็นเจ้าของเครื่องมือส่วนนี้เอง แทนที่จะต้องยืมคนอื่น

## เอกสารอ้างอิง

[1] S. Sanghlao and S. Sanglao, "Agentic LLM-Driven RTL Design for Factor-Graph Processing Elements Using a Containerized EDA Loop," *Proc. APSIPA ASC 2026*, to appear.
