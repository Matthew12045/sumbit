# mechanical-rules

## mechanical

### mechanical/connective-stack

**Rule:** Use the formal connective stack (การที่...ทำให, การ...ส่งผลให้, ผ่าน..., ผ่านการ...)
sparingly. AI inserts these where a simple space or direct statement works.

**Bad:** `การที่ระบบประมวลผลข้อมูลเร็วขึ้นส่งผลให้ทีมงานสามารถทำงานได้มากขึ้น`
**Good:** `ระบบประมวลผลข้อมูลเร็วขึ้น ทีมงานทำงานได้มากขึ้น`

**Why:** Formal connective stack is a direct calque of English causal chains ("The fact that X causes Y to happen"). Native Thai prefers juxtaposition or simple `เพราะ...จึง...`.

**When it fires:** When you see `การที่` + noun phrase + `ส่งผลให้` / `ทำให้` chains.

---

### mechanical/passive-voice

**Rule:** Avoid passive voice (`ถูก`, `ได้รับ`, `ถูกทำให้`) unless the agent is genuinely unknown or the patient is the focus. Thai prefers active voice with topic-fronting.

**Bad:** `ข้อมูลถูกประมวลผลโดยระบบทุก 5 นาที`
**Good:** `ระบบประมวลผลข้อมูลทุก 5 นาที` / `ข้อมูล ระบบประมวลผลทุก 5 นาที`

**Why:** Thai has a `ถูก` passive but it carries adversative/beneficiary meaning. Neutral passives are calques.

**When it fires:** `ถูก` + verb where the agent follows or is obvious.

---

### mechanical/nominalization

**Rule:** Avoid turning verbs into nouns just to add `การ` or `ความ` prefixes. Thai can use verbs directly.

**Bad:** `เรา需要进行วิเคราะห์ข้อมูล` (we need to do analysis of data)
**Good:** `เราต้องวิเคราะห์ข้อมูล` (we must analyze data)

**Why:** Nominalization is an English habit — "we need to do an analysis of" → "เรา需要进行วิเคราะห์". Thai doesn't need the noun wrapper.

**When it fires:** `การ`/`ความ` + verb where the verb could stand alone.

---

### mechanical/panorama-openers

**Rule:** Don't open paragraphs with panorama phrases (`ในโลกที่...`, `ในยุคที่...`, `ในปัจจุบันที่...`). These are filler openers that add no information.

**Bad:** `ในยุคที่เทคโนโลยีพัฒนาอย่างรวดเร็ว ระบบนี้ช่วยคุณ...`
**Good:** `เทคโนโลยีพัฒนาเร็ว ระบบนี้ช่วยคุณ...`

**Why:** Panorama openers are padding — they sound formal but say nothing. Native Thai writers jump straight into the topic.

**When it fires:** Paragraph opening with `ใน...ที่` / `ในยุคที่` / `ในโลกที่`.

---

### mechanical/padding

**Rule:** Remove padding phrases that pad length without adding meaning: `อย่างทราบกันดีว่า`, `ต้องยอมรับว่า`, `อย่างไม่ต้องสงสัย`, `อย่างชัดเจน`.

**Bad:** `อย่างไม่ต้องสงสัยว่าระบบนี้ช่วยให้工作效率เพิ่มขึ้น`
**Good:** `ระบบนี้ช่วยเพิ่มประสิทธิภาพการทำงาน`

**Why:** These are hedge phrases that AI adds to sound formal. They're empty calories.

**When it fires:** Common padding phrases at the start of sentences.

---

### mechanical/pronoun-scam

**Rule:** Don't overuse pronouns (`มัน`, `เขา`, `พวกเขา`, `สิ่งนี้`, `สิ่งนั้น`). Thai uses zero anaphora — once a topic is set, the subject can be dropped entirely.

**Bad:** `ระบบนี้ช่วยจัดการข้อมูล มันช่วยให้ทีมงานทำงานได้เร็วขึ้น มันยังลดข้อผิดพลาดด้วย`
**Good:** `ระบบนี้ช่วยจัดการข้อมูล ทีมงานทำงานได้เร็วขึ้น ลดข้อผิดพลาดด้วย`

**Why:** Every pronoun is a missed opportunity for zero anaphora. Each unnecessary pronoun makes the text sound translated.

**When it fires:** Consecutive clauses each starting with a pronoun where the subject hasn't changed.

---

### mechanical/punctuation-rules

**Rule:** 
- No period spam — mid-paragraph periods should be spaces.
- No English-style quotation marks ("...") for Thai — use `"...".` or just plain text.
- Use Thai quotation marks (`"..."`) when quotes are needed.
- Ellipsis is `...` (three dots), not `…` (single ellipsis char) or `...` with spaces.

**Bad:** `ระบบทำงานเร็วขึ้น. ใช้ memory น้อยลง. ทีมงานพอใจมาก.`
**Good:** `ระบบทำงานเร็วขึ้น ใช้ memory น้อยลง ทีมงานพอใจมาก`

**Why:** Period spam is the most visible AI tell in Thai text. It's a direct calque of English sentence boundary marking.

**When it fires:** Multiple periods within a single paragraph that don't mark true sentence boundaries.

---

## References

- `grammar.md` — surface grammar rules (classifiers, modals, calques)
- `style-rules.md` — positive style rules
- `craft.md` — soft taste rules
- `register.md` — register-specific rules
- `examples.md` — worked before/after examples
