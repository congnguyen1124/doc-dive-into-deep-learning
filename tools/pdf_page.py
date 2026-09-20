#!/usr/bin/env python3
"""Extract one physical page of a PDF into a standalone, optionally upscaled 1-page PDF.\n\nUpscaling lets a 72-DPI rasterizer such as macOS `sips` produce a high-resolution\nrender without a dedicated PDF rendering library."""
from __future__ import annotations
import sys, zlib
from pdf_reader import PDF, Ref, Stream


class Writer:
    def __init__(self, doc: PDF, scale: float = 1.0):
        self.doc = doc
        self.scale = scale
        self.objects = []        # 1-based list of serialized bodies
        self.map = {}            # old object number -> new number

    def alloc(self) -> int:
        self.objects.append(None)
        return len(self.objects)

    def put(self, num: int, body: bytes):
        self.objects[num - 1] = body

    def copy(self, num: int) -> int:
        """Copy object `num` (and everything it references) from the source document."""
        if num in self.map:
            return self.map[num]
        new = self.alloc()
        self.map[num] = new
        obj = self.doc.get(num)
        self.put(new, self.ser(obj))
        return new

    def ser(self, obj) -> bytes:
        d = self.doc
        if isinstance(obj, Ref):
            return b"%d 0 R" % self.copy(obj.num)
        if isinstance(obj, Stream):
            body = obj.raw
            head = dict(obj.dict)
            head["Length"] = len(body)
            return self.ser(head) + b"\nstream\n" + body + b"\nendstream"
        if isinstance(obj, bool):
            return b"true" if obj else b"false"
        if isinstance(obj, int):
            return b"%d" % obj
        if isinstance(obj, float):
            return (b"%.6f" % obj).rstrip(b"0").rstrip(b".") or b"0"
        if isinstance(obj, bytes):
            return b"<" + obj.hex().encode("ascii") + b">"
        if isinstance(obj, str):
            return b"/" + obj.encode("latin-1")
        if isinstance(obj, list):
            return b"[" + b" ".join(self.ser(x) for x in obj) + b"]"
        if isinstance(obj, dict):
            parts = [b"/" + k.encode("latin-1") + b" " + self.ser(v) for k, v in obj.items()]
            return b"<<" + b" ".join(parts) + b">>"
        return b"null"

    def build(self, page: dict) -> bytes:
        d = self.doc
        media = [float(d.resolve(x)) for x in (d.resolve(page.get("MediaBox")) or [0, 0, 612, 792])]
        s = self.scale
        scaled = [media[0] * s, media[1] * s, media[2] * s, media[3] * s]

        # concatenate the page's content streams and prepend the scaling matrix
        contents = d.resolve(page.get("Contents"))
        chunks = []
        for c in (contents if isinstance(contents, list) else [contents]):
            c = d.resolve(c)
            if isinstance(c, Stream):
                chunks.append(c.data())
        raw = b"q %f 0 0 %f 0 0 cm\n" % (s, s) + b"\n".join(chunks) + b"\nQ"
        packed = zlib.compress(raw)

        content_num = self.alloc()
        self.put(content_num, self.ser(Stream({"Filter": "FlateDecode"}, packed, d)))

        page_num = self.alloc()
        pages_num = self.alloc()
        root_num = self.alloc()

        new_page = {
            "Type": "Page",
            "Parent": Ref(-pages_num, 0),
            "MediaBox": scaled,
            "Contents": Ref(-content_num, 0),
        }
        res = page.get("Resources")
        if res is not None:
            new_page["Resources"] = res if isinstance(res, Ref) else res
        # serialize with already-new refs passed through
        self.put(page_num, self._ser_new(new_page))
        self.put(pages_num, self._ser_new({"Type": "Pages", "Kids": [Ref(-page_num, 0)], "Count": 1}))
        self.put(root_num, self._ser_new({"Type": "Catalog", "Pages": Ref(-pages_num, 0)}))

        out = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0] * (len(self.objects) + 1)
        for i, body in enumerate(self.objects, start=1):
            offsets[i] = len(out)
            out += b"%d 0 obj\n" % i + (body or b"null") + b"\nendobj\n"
        xref_pos = len(out)
        n = len(self.objects) + 1
        out += b"xref\n0 %d\n" % n
        out += b"0000000000 65535 f \n"
        for i in range(1, n):
            out += b"%010d 00000 n \n" % offsets[i]
        out += b"trailer\n" + self._ser_new({"Size": n, "Root": Ref(-root_num, 0)})
        out += b"\nstartxref\n%d\n%%%%EOF\n" % xref_pos
        return bytes(out)

    def _ser_new(self, obj) -> bytes:
        """Serialize, treating Ref with a negative num as an already-allocated new object."""
        if isinstance(obj, Ref) and obj.num < 0:
            return b"%d 0 R" % (-obj.num)
        if isinstance(obj, dict):
            parts = [b"/" + k.encode("latin-1") + b" " + self._ser_new(v) for k, v in obj.items()]
            return b"<<" + b" ".join(parts) + b">>"
        if isinstance(obj, list):
            return b"[" + b" ".join(self._ser_new(x) for x in obj) + b"]"
        return self.ser(obj)


def main():
    pdf_path, page_no, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    doc = PDF(pdf_path)
    page = doc.pages()[page_no - 1]
    data = Writer(doc, scale).build(page)
    open(out_path, "wb").write(data)
    print(f"wrote {out_path} ({len(data)} bytes, scale {scale})")


if __name__ == "__main__":
    main()
