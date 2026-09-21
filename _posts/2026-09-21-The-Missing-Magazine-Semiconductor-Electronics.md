---
layout: post
title:  "The Missing Magazine: วารสารเซมิคอนดักเตอร์ อิเล็กทรอนิกส์ and the Layer Thai Electronics Lost"
date:   2026-09-21 16:00:00 +0700
categories: Electronics Education
tags: [semiconductor-electronics, se-ed, hobby-electronics, maker, thailand, ai, aliexpress, shopee, iot]
---

*[ฉบับภาษาไทย](/posts/The-Missing-Magazine-Semiconductor-Electronics-TH/)*

Ask an older Thai engineer how they got into electronics, and many will name a magazine before they name a university: **วารสารเซมิคอนดักเตอร์ อิเล็กทรอนิกส์ (Semiconductor Electronics Journal)**. It came out for more than four decades. Its print edition stopped in 2019. Almost nobody noticed, because by then a module from AliExpress cost less than the magazine.

This post covers what the magazine was, what it did that nothing does today, and why that gap matters more now that AI writes our datasheets.

## What the magazine was

The magazine isn't a side note in SE-ED's history. It is the reason SE-ED exists.

| When | What | Source |
|---|---|---|
| 10 Oct 1974 (2517) | บริษัท ซีเอ็ดยูเคชั่น registered with ฿99,000 capital by **ten electrical engineers from Chulalongkorn's Faculty of Engineering**. Its first businesses: the monthly magazine *เซมิคอนดัคเตอร์อิเล็กทรอนิกส์* and importing electronic components. | SE-ED 56-1 filing; Thai Wikipedia |
| 1980s | The magazine sat at the centre of a family of titles: *ทักษะ*, *มิติที่ 4*, *ไมโครคอมพิวเตอร์* (1983), and later *รู้รอบตัว / UpDATE*. | Thai Wikipedia |
| Oct–Dec 1982 (2525) | Issue **No. 53** covered antennas and the "golden age of microcomputers". Collectors still sell copies of it. | booknarts.com listing |
| 1991 onward | SE-ED listed on the SET and opened the SE-ED Book Center chain. Over time the bookstore business outgrew the magazine that started it. | Thai Wikipedia |
| 2004–2007 | Four years of back issues re-released on CD-ROM, which university libraries still hold | PSU Trang OPAC |
| 2008 | Renamed **Semiconductor Electronics Plus+**. Issues ~382–423 were also sold as PDF e-magazines. | se-ed.com e-magazine listings |
| 2019 (2562) | **Print edition ceased.** University library notices give the month as anywhere from March to September 2562. | Thai university library notices |

ISSN **0125-1015**. More than 45 years, over 400 issues.

A small group of engineers started a magazine to teach Thai people electronics, sold parts next to it, and grew it into one of the country's biggest book retailers. The educational mission came first and retail came second. Eventually the retail side was all that was left.

## What it did that nothing replaced

Last week I argued that US electronics didn't grow because of RadioShack alone. It grew because of a connected chain: **parts → knowledge → community → industry**. In the US, *Popular Electronics* was the knowledge link. The Altair 8800 on its January 1975 cover brought the Homebrew Computer Club together.

For Thailand, *Semiconductor Electronics* was that link, and it did three things at once:

1. **It was curated and edited.** A circuit in print had been built, checked and proofread by someone whose name was on it. Errors came back as letters, and the corrections appeared in the next issue.
2. **It was in Thai and tied to Thai parts.** Projects used components you could actually buy at บ้านหม้อ, often through the same company that printed the magazine. The article and the parts came from one place.
3. **It came with a community.** Readers wrote in, sent projects and argued about designs. For a kid in the provinces, it was the Homebrew Computer Club arriving by post once a month.

When print stopped, **access** didn't disappear. Parts got cheaper and information multiplied. What disappeared was the **curated, Thai-language, accountable layer** between a component and a learner.

## What filled the gap

| Layer | The magazine era | 2026 |
|---|---|---|
| Parts | บ้านหม้อ and SE-ED's own imports; slow and expensive | AliExpress, Shopee, Lazada: an ESP32 for about ฿150, delivered in days ✅ |
| Prototyping | Etching your own boards with ferric chloride | 5 PCBs for about $2 from China ✅ |
| Knowledge | Edited, tested, in Thai, with named authors | YouTube, forums, and increasingly **AI-generated pages with no author** ⚠️ |
| Correction loop | Reader letters, then an erratum in the next issue | None. Wrong pages stay up and rank well. ❌ |
| Community | Readers across the whole country, one shared reference | Scattered Facebook groups and LINE chats ⚠️ |

Two layers got dramatically better and one got dramatically worse.

### A concrete example

While working on an IoT audio project, I looked up the **INMP441** microphone on a well-known circuit-design site. The page:

- called it "INM441" in the title and "INMP441" in the body
- described it as a **2-pin analog electret capsule** with a 1.5–10 V supply
- wired it through a 10 µF capacitor into `analogRead(A0)` on an Arduino Uno

The real INMP441 is a **6-pin digital I²S MEMS microphone** (VDD, GND, SD, SCK, WS, L/R) that runs on 1.8–3.3 V. A classic Uno has no I²S peripheral at all. Nearly every line of the page is wrong, yet it reads fluently, looks professional and has no author or erratum.

An edited magazine would never have printed that page. If it had somehow slipped through, a reader would have written in. Today the error just sits there, and the next AI model trains on it.

## Can AliExpress, Shopee and AI replace it?

Partly. AI has one ability the magazine never had: **it answers your exact question in Thai at 2 a.m.** As a tutor it beats any monthly publication.

A tutor is not an editor, though. What we lost was not information. We lost **accountability**: a named person who tested the circuit, a place to report errors, and a shared reference that a whole generation learned from. Cheap parts plus fluent AI produce **consumers who can copy**, while the magazine produced **builders who could check**.

## Rebuilding the missing layer

We don't need to bring back print. We need to bring back what the magazine did:

1. **Curated, tested, Thai-language project notes.** Put them in a Git repository or a blog like this one, and require every circuit to be built, photographed and measured before it's published.
2. **A visible correction loop.** Issues and pull requests are today's reader letters. Keep errata public.
3. **Datasheet first, AI second.** Teach students to check an AI answer against the manufacturer datasheet. Pin count, supply voltage and interface type are the three checks that would have caught the INMP441 page.
4. **Link the article to the parts.** The magazine sold the parts it wrote about. Today that means a verified Shopee or AliExpress part list for each project, noting which listings actually shipped the right chip.
5. **Universities as publishers.** SE-ED started with ten Chula engineering graduates. Thai universities have hundreds of electronics students each year whose lab reports could become that tested, public layer, instead of being thrown away after grading.

## Closing

Ten engineers in 1974 decided Thai people should be able to learn electronics in Thai from a source they could trust. The company they started is still here, but the magazine isn't. The parts are cheaper than ever and the AI answers fluently. What's missing is someone who checks.

That job doesn't need a publishing house anymore. It needs a few people willing to build a circuit before they post it, and to fix it publicly when they're wrong.

---

### Sources and notes

- SE-ED Education PCL, Form 56-1 (company history: registered 10 Oct 2517, ฿99,000, the monthly *เซมิคอนดัคเตอร์อิเล็กทรอนิกส์* and component distribution). [corporate.se-ed.com](http://corporate.se-ed.com/wp-content/uploads/2017/09/SE-ED_56-1_Form-2542_TH.pdf)
- ซีเอ็ดยูเคชั่น, Thai Wikipedia (magazine timeline, SET listing, bookstores). [th.wikipedia.org](https://th.wikipedia.org/wiki/%E0%B8%8B%E0%B8%B5%E0%B9%80%E0%B8%AD%E0%B9%87%E0%B8%94%E0%B8%A2%E0%B8%B9%E0%B9%80%E0%B8%84%E0%B8%8A%E0%B8%B1%E0%B9%88%E0%B8%99)
- Brandcase, "กรณีศึกษา SE-ED" (ten Chula engineering graduates, the early specialist-book business). [brandcase.co](https://www.brandcase.co/42776)
- SE-ED product page for the journal (ISSN barcode 977-0125-101-00-5). [m.se-ed.com](https://m.se-ed.com/Detail/วารสาร-เซมิคอนดักเตอร์-อิเล็กทรอนิกส์-(Semiconductor-Electronics-Journal)/9770125101005)
- *Semiconductor Electronics Plus+* e-magazine listings (issues 382–423 as PDF). [se-ed.com](https://www.se-ed.com/product-magazine-code/e-magazine/วารสาร-นิตยสารในเครือซีเอ็ด/semiconductor-electronics-plus.aspx?nc=E5859&mid=4112&mc=0000002)
- Issue No. 53, Oct–Dec 2525, collector listing. [booknarts.com](http://www.booknarts.com/product/1488/)
- CD-ROM compilation 2547–2550, PSU Trang library. [opac.trang.psu.ac.th](https://opac.trang.psu.ac.th/BibDetail.aspx?bibno=22425)
- Silpakorn University library, "วารสารที่หอสมุดฯ บอกรับและหยุดพิมพ์ ในเดือนเมษายน 2562". [snamcn.lib.su.ac.th](http://www.snamcn.lib.su.ac.th/snclibblog/?p=68639)

*Verification note: The name change to Plus+ (2008) and the end of print (2019) come from library-catalogue summaries. The exact final month and the last issue number are unconfirmed, because the sources disagree (March, April and September 2562 all appear). If you have the last issue on your shelf, please tell me the number and I'll correct this post. That is the correction loop this post argues for.*
