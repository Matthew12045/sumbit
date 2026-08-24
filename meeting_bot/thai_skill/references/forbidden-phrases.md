# forbidden-phrases.md

## Blocklist of AI-generated Thai phrases

This is an audit blocklist. If these phrases appear in output, they should be
flagged and replaced with more natural Thai.

### Core forbidden phrases

| AI phrase | Why | Replace with |
|-----------|-----|-------------|
| โดยสรุปแล้ว | Recap close — restates the body | Drop it, end forward-looking |
| อย่างไรก็ตาม | Formal connective calque | `แต` or just start the sentence |
| นอกจากนี้ | Formal connective calque | `ยัง` or `และ` |
| ทั้งนี | Filler, adds nothing | Drop entirely |
| จึงขอสรุปว่า | Recap close | Drop |
| อย่างยิ่ง | Empty intensifier | Be specific |
| อย่างมาก | Empty intensifier | Be specific |
| อย่างมีนัยสำคัญ | Vague intensifier | Give numbers |
| ในยุคดิจิทัล | Panorama filler | Jump to the topic |
| ในโลกออนไลน์ | Panorama filler | Jump to the topic |
| ในยุคที่ | Panorama filler | Jump to the topic |
| เป็นที่ทราบกันดีว่า | Padding | Drop |
| ต้องยอมรับว่า | Padding | Drop |
| อย่างไม่ต้องสงสัย | Padding | Drop |
| อย่างชัดเจน | Padding | Drop |
| ปฏิวัติ | Cliche headline | Concrete verb |
| transformación | Cliche headline | Concrete verb |
| ก้าวใหม่ | Cliche headline | Concrete statement |
| ยุคใหม่ | Cliche headline | Concrete statement |
| จุดเปลี่ยน | Cliche headline | Concrete statement |
| หมดปัญหา | Generic reassurance | What actually happens |
| ไว้วางใจได้ | Generic reassurance | What makes it reliable |
| ช่วยให้คุณ | Overuse of `ช่วย` | Direct verb |
| ซึ่งเป็นที่ | Connective stacking | Start new sentence |
| ซึ่งทำให้ | Connective stacking | Start new sentence |
| ส่งผลให้ | Connective stacking | Start new sentence |

### Thai-specific AI tells

| AI phrase | Why |
|-----------|-----|
| การที่...ส่งผลให้ | SVO calque chain |
| การ...ทำให้ | Nominalization + causative chain |
| ซึ่ง... | Relative clause stacking |
| นั้นคือ | Calque of "that is" |
| นั่นคือเหตุผลว่าทำไม | Calque of "that's why" |
| เหมือนกับว่า | Calque of "as if" |
| ในขณะเดียวกัน | Formal connective |
| ในทางกลับกัน | Formal connective |
| ในขณะเดียวกันนั้น | Double formal connective |

## References

- `ai-tells.md` — mechanical rules
- `examples.md` — worked examples showing replacements
