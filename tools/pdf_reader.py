#!/usr/bin/env python3
"""Minimal pure-python PDF reader: parses PDF 1.5 (xref streams + object streams)\nand extracts page text. Used by the Poppler-free figure extractor."""
from __future__ import annotations
import re, zlib, sys

WS = b"\x00\t\n\x0c\r "
DELIM = b"()<>[]{}/%"


class Lexer:
    def __init__(self, buf: bytes, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def skip_ws(self):
        b = self.buf
        n = len(b)
        while self.pos < n:
            c = b[self.pos]
            if c in WS:
                self.pos += 1
            elif c == 0x25:  # %
                while self.pos < n and b[self.pos] not in b"\r\n":
                    self.pos += 1
            else:
                return

    def token(self):
        self.skip_ws()
        b = self.buf
        n = len(b)
        if self.pos >= n:
            return None
        c = b[self.pos]
        if c == 0x2F:  # /
            start = self.pos
            self.pos += 1
            while self.pos < n and b[self.pos] not in WS and b[self.pos] not in DELIM:
                self.pos += 1
            return ("name", b[start + 1:self.pos].decode("latin-1"))
        if c == 0x28:  # (
            return ("str", self.read_lit_string())
        if c == 0x3C:  # <
            if self.pos + 1 < n and b[self.pos + 1] == 0x3C:
                self.pos += 2
                return ("op", "<<")
            return ("str", self.read_hex_string())
        if c == 0x3E:  # >
            if self.pos + 1 < n and b[self.pos + 1] == 0x3E:
                self.pos += 2
                return ("op", ">>")
            self.pos += 1
            return ("op", ">")
        if c in b"[]{}":
            self.pos += 1
            return ("op", chr(c))
        start = self.pos
        while self.pos < n and b[self.pos] not in WS and b[self.pos] not in DELIM:
            self.pos += 1
        if self.pos == start:
            self.pos += 1
        raw = b[start:self.pos]
        try:
            if re.fullmatch(rb"[+-]?\d+", raw):
                return ("num", int(raw))
            if re.fullmatch(rb"[+-]?(\d*\.\d*|\d+)", raw):
                return ("num", float(raw))
        except ValueError:
            pass
        return ("kw", raw.decode("latin-1"))

    def read_lit_string(self) -> bytes:
        b = self.buf
        n = len(b)
        self.pos += 1
        depth = 1
        out = bytearray()
        while self.pos < n:
            c = b[self.pos]
            if c == 0x5C:  # backslash
                self.pos += 1
                if self.pos >= n:
                    break
                e = b[self.pos]
                mapping = {0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12}
                if e in mapping:
                    out.append(mapping[e])
                    self.pos += 1
                elif 0x30 <= e <= 0x37:
                    oct_digits = b""
                    while self.pos < n and len(oct_digits) < 3 and 0x30 <= b[self.pos] <= 0x37:
                        oct_digits += b[self.pos:self.pos + 1]
                        self.pos += 1
                    out.append(int(oct_digits, 8) & 0xFF)
                elif e in b"\r\n":
                    self.pos += 1
                    if self.pos < n and b[self.pos] in b"\n" and e == 0x0D:
                        self.pos += 1
                else:
                    out.append(e)
                    self.pos += 1
                continue
            if c == 0x28:
                depth += 1
            elif c == 0x29:
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    break
            out.append(c)
            self.pos += 1
        return bytes(out)

    def read_hex_string(self) -> bytes:
        b = self.buf
        self.pos += 1
        end = b.index(b">", self.pos)
        hexs = re.sub(rb"[^0-9A-Fa-f]", b"", b[self.pos:end])
        self.pos = end + 1
        if len(hexs) % 2:
            hexs += b"0"
        return bytes.fromhex(hexs.decode("ascii"))


class Ref:
    __slots__ = ("num", "gen")

    def __init__(self, num, gen):
        self.num, self.gen = num, gen

    def __repr__(self):
        return f"Ref({self.num},{self.gen})"


class Stream:
    __slots__ = ("dict", "raw", "doc")

    def __init__(self, d, raw, doc):
        self.dict, self.raw, self.doc = d, raw, doc

    def data(self) -> bytes:
        filters = self.doc.resolve(self.dict.get("Filter"))
        if filters is None:
            return self.raw
        if not isinstance(filters, list):
            filters = [filters]
        data = self.raw
        parms = self.doc.resolve(self.dict.get("DecodeParms"))
        if not isinstance(parms, list):
            parms = [parms] * len(filters)
        for f, p in zip(filters, parms):
            if f in ("FlateDecode", "Fl"):
                try:
                    data = zlib.decompress(data)
                except zlib.error:
                    data = zlib.decompressobj().decompress(data)
                p = self.doc.resolve(p)
                if isinstance(p, dict) and self.doc.resolve(p.get("Predictor", 1)) >= 2:
                    data = apply_png_predictor(
                        data,
                        int(self.doc.resolve(p.get("Colors", 1))),
                        int(self.doc.resolve(p.get("BitsPerComponent", 8))),
                        int(self.doc.resolve(p.get("Columns", 1))),
                    )
            elif f == "ASCIIHexDecode":
                data = bytes.fromhex(re.sub(rb"[^0-9A-Fa-f]", b"", data.split(b">")[0]).decode())
            else:
                pass
        return data


def apply_png_predictor(data: bytes, colors: int, bpc: int, columns: int) -> bytes:
    bpp = max(1, (colors * bpc + 7) // 8)
    rowlen = (columns * colors * bpc + 7) // 8
    out = bytearray()
    prev = bytearray(rowlen)
    i = 0
    n = len(data)
    while i + 1 <= n - 1:
        ft = data[i]
        i += 1
        row = bytearray(data[i:i + rowlen])
        if len(row) < rowlen:
            row.extend(b"\x00" * (rowlen - len(row)))
        i += rowlen
        if ft == 1:
            for j in range(bpp, rowlen):
                row[j] = (row[j] + row[j - bpp]) & 0xFF
        elif ft == 2:
            for j in range(rowlen):
                row[j] = (row[j] + prev[j]) & 0xFF
        elif ft == 3:
            for j in range(rowlen):
                left = row[j - bpp] if j >= bpp else 0
                row[j] = (row[j] + ((left + prev[j]) >> 1)) & 0xFF
        elif ft == 4:
            for j in range(rowlen):
                a = row[j - bpp] if j >= bpp else 0
                b_ = prev[j]
                c = prev[j - bpp] if j >= bpp else 0
                p = a + b_ - c
                pa, pb, pc = abs(p - a), abs(p - b_), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b_ if pb <= pc else c)
                row[j] = (row[j] + pr) & 0xFF
        out.extend(row)
        prev = row
    return bytes(out)


class Parser(Lexer):
    def __init__(self, buf, pos, doc):
        super().__init__(buf, pos)
        self.doc = doc

    def parse(self):
        tok = self.token()
        return self._from(tok)

    def _from(self, tok):
        if tok is None:
            return None
        kind, val = tok
        if kind in ("str", "name"):
            return val if kind == "name" else val
        if kind == "num":
            save = self.pos
            t2 = self.token()
            if t2 and t2[0] == "num" and isinstance(val, int) and isinstance(t2[1], int):
                save2 = self.pos
                t3 = self.token()
                if t3 and t3[0] == "kw" and t3[1] == "R":
                    return Ref(val, t2[1])
                self.pos = save
                return val
            self.pos = save
            return val
        if kind == "op":
            if val == "[":
                arr = []
                while True:
                    save = self.pos
                    t = self.token()
                    if t is None or (t[0] == "op" and t[1] == "]"):
                        return arr
                    self.pos = save
                    arr.append(self.parse())
            if val == "<<":
                d = {}
                while True:
                    t = self.token()
                    if t is None or (t[0] == "op" and t[1] == ">>"):
                        break
                    if t[0] != "name":
                        continue
                    d[t[1]] = self.parse()
                save = self.pos
                t = self.token()
                if t and t[0] == "kw" and t[1] == "stream":
                    b = self.buf
                    p = self.pos
                    if b[p:p + 2] == b"\r\n":
                        p += 2
                    elif b[p:p + 1] in (b"\n", b"\r"):
                        p += 1
                    length = self.doc.resolve(d.get("Length")) if self.doc else d.get("Length")
                    if not isinstance(length, int):
                        e = b.find(b"endstream", p)
                        length = e - p
                    raw = b[p:p + length]
                    e = b.find(b"endstream", p + length)
                    if e == -1:
                        e = p + length
                    self.pos = e + len(b"endstream")
                    return Stream(d, raw, self.doc)
                self.pos = save
                return d
            return None
        if kind == "kw":
            if val == "true":
                return True
            if val == "false":
                return False
            if val == "null":
                return None
            return ("kw", val)
        return None


class PDF:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.xref = {}
        self.compressed = {}
        self.trailer = {}
        self.cache = {}
        self._objstm_cache = {}
        self._load_xref()

    def _load_xref(self):
        idx = self.buf.rfind(b"startxref")
        m = re.search(rb"startxref\s+(\d+)", self.buf[idx:])
        start = int(m.group(1))
        seen = set()
        while start and start not in seen:
            seen.add(start)
            start = self._read_xref_section(start)

    def _read_xref_section(self, pos):
        p = Parser(self.buf, pos, self)
        p.skip_ws()
        if self.buf[p.pos:p.pos + 4] == b"xref":
            p.pos += 4
            while True:
                p.skip_ws()
                if self.buf[p.pos:p.pos + 7] == b"trailer":
                    p.pos += 7
                    tr = p.parse()
                    for k, v in (tr or {}).items():
                        self.trailer.setdefault(k, v)
                    nxt = tr.get("XRefStm")
                    if isinstance(nxt, int):
                        self._read_xref_section(nxt)
                    prev = tr.get("Prev")
                    return prev if isinstance(prev, int) else None
                m = re.match(rb"(\d+)\s+(\d+)", self.buf[p.pos:p.pos + 40])
                if not m:
                    return None
                first, count = int(m.group(1)), int(m.group(2))
                p.pos += m.end()
                p.skip_ws()
                for i in range(count):
                    entry = self.buf[p.pos:p.pos + 20]
                    em = re.match(rb"(\d{10})\s(\d{5})\s([nf])", entry)
                    if em:
                        if em.group(3) == b"n":
                            self.xref.setdefault(first + i, int(em.group(1)))
                        p.pos += 20 if entry[18:20] in (b"\r\n", b" \n", b" \r") else em.end()
                        p.skip_ws() if entry[18:20] not in (b"\r\n", b" \n", b" \r") else None
                    else:
                        return None
        # xref stream
        m = re.match(rb"\s*(\d+)\s+(\d+)\s+obj", self.buf[pos:pos + 64])
        if not m:
            return None
        p = Parser(self.buf, pos + m.end(), self)
        stm = p.parse()
        if not isinstance(stm, Stream):
            return None
        d = stm.dict
        for k, v in d.items():
            self.trailer.setdefault(k, v)
        w = [int(self.resolve(x)) for x in self.resolve(d["W"])]
        size = int(self.resolve(d["Size"]))
        index = self.resolve(d.get("Index")) or [0, size]
        index = [int(self.resolve(x)) for x in index]
        data = stm.data()
        rowlen = sum(w)
        off = 0
        for k in range(0, len(index), 2):
            first, count = index[k], index[k + 1]
            for i in range(count):
                if off + rowlen > len(data):
                    break
                fields = []
                q = off
                for width in w:
                    fields.append(int.from_bytes(data[q:q + width], "big") if width else None)
                    q += width
                off += rowlen
                ftype = fields[0] if w[0] else 1
                num = first + i
                if ftype == 1 and num not in self.xref and num not in self.compressed:
                    self.xref[num] = fields[1]
                elif ftype == 2 and num not in self.xref and num not in self.compressed:
                    self.compressed[num] = (fields[1], fields[2])
        prev = self.resolve(d.get("Prev"))
        return prev if isinstance(prev, int) else None

    def resolve(self, obj, depth=0):
        while isinstance(obj, Ref) and depth < 64:
            obj = self.get(obj.num)
            depth += 1
        return obj

    def get(self, num):
        if num in self.cache:
            return self.cache[num]
        self.cache[num] = None
        val = None
        if num in self.xref:
            pos = self.xref[num]
            m = re.match(rb"\s*(\d+)\s+(\d+)\s+obj", self.buf[pos:pos + 64])
            if m:
                p = Parser(self.buf, pos + m.end(), self)
                val = p.parse()
        elif num in self.compressed:
            stm_num, idx = self.compressed[num]
            table = self._objstm(stm_num)
            val = table.get(num)
        self.cache[num] = val
        return val

    def _objstm(self, stm_num):
        if stm_num in self._objstm_cache:
            return self._objstm_cache[stm_num]
        self._objstm_cache[stm_num] = {}
        stm = self.resolve(Ref(stm_num, 0))
        if not isinstance(stm, Stream):
            return {}
        data = stm.data()
        n = int(self.resolve(stm.dict["N"]))
        first = int(self.resolve(stm.dict["First"]))
        header = data[:first].split()
        table = {}
        for i in range(n):
            try:
                onum = int(header[2 * i])
                ooff = int(header[2 * i + 1])
            except (IndexError, ValueError):
                break
            table[onum] = Parser(data, first + ooff, self).parse()
        self._objstm_cache[stm_num] = table
        return table

    def pages(self):
        root = self.resolve(self.trailer.get("Root"))
        tree = self.resolve(root["Pages"])
        out = []

        def walk(node, inherited):
            node = self.resolve(node)
            if not isinstance(node, dict):
                return
            inh = dict(inherited)
            for k in ("Resources", "MediaBox", "Rotate"):
                if k in node:
                    inh[k] = node[k]
            if self.resolve(node.get("Type")) == "Page" or ("Kids" not in node and "Contents" in node):
                merged = dict(inh)
                merged.update(node)
                out.append(merged)
                return
            for kid in self.resolve(node.get("Kids")) or []:
                walk(kid, inh)

        walk(tree, {})
        return out


# ---------------- font / text ----------------

STD_ENC_EXTRA = {
    0o20: "ı", 0o21: "ȷ", 0o22: "`", 0o23: "´", 0o24: "ˇ",
    0o25: "˘", 0o26: "¯", 0o27: "˚", 0o30: "¸", 0o31: "ß",
    0o32: "æ", 0o33: "œ", 0o34: "ø", 0o35: "Æ", 0o36: "Œ",
    0o37: "Ø",
}


def parse_tounicode(data: bytes) -> dict:
    out = {}
    text = data.decode("latin-1", "replace")
    for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            out[int(src, 16)] = decode_utf16be(dst)
    for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
        for m in re.finditer(
            r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(?:<([0-9A-Fa-f]+)>|\[(.*?)\])", block, re.S
        ):
            lo, hi = int(m.group(1), 16), int(m.group(2), 16)
            if m.group(3):
                base = m.group(3)
                for i in range(hi - lo + 1):
                    try:
                        val = int(base, 16) + i
                        out[lo + i] = chr(val) if val < 0x110000 else ""
                    except ValueError:
                        pass
            elif m.group(4):
                items = re.findall(r"<([0-9A-Fa-f]+)>", m.group(4))
                for i, it in enumerate(items):
                    out[lo + i] = decode_utf16be(it)
    return out


def decode_utf16be(h: str) -> str:
    try:
        b = bytes.fromhex(h if len(h) % 2 == 0 else h + "0")
        return b.decode("utf-16-be", "replace")
    except ValueError:
        return ""


GLYPH_RE = re.compile(r"^(?:uni([0-9A-Fa-f]{4})|u([0-9A-Fa-f]{4,6}))$")
GLYPH_NAMES = {
    "space": " ", "exclam": "!", "quotedbl": '"', "numbersign": "#", "dollar": "$",
    "percent": "%", "ampersand": "&", "quoteright": "'", "quotesingle": "'",
    "parenleft": "(", "parenright": ")", "asterisk": "*", "plus": "+", "comma": ",",
    "hyphen": "-", "period": ".", "slash": "/", "zero": "0", "one": "1", "two": "2",
    "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "colon": ":", "semicolon": ";", "less": "<", "equal": "=",
    "greater": ">", "question": "?", "at": "@", "bracketleft": "[", "backslash": "\\",
    "bracketright": "]", "asciicircum": "^", "underscore": "_", "quoteleft": "`",
    "braceleft": "{", "bar": "|", "braceright": "}", "asciitilde": "~",
    "endash": "–", "emdash": "—", "quotedblleft": "“",
    "quotedblright": "”", "quotedblbase": "„", "bullet": "•",
    "fi": "fi", "fl": "fl", "ff": "ff", "ffi": "ffi", "ffl": "ffl",
    "dotlessi": "ı", "germandbls": "ß", "ae": "æ", "oe": "œ",
}


def glyph_to_char(name: str) -> str:
    if name in GLYPH_NAMES:
        return GLYPH_NAMES[name]
    m = GLYPH_RE.match(name)
    if m:
        code = int(m.group(1) or m.group(2), 16)
        return chr(code) if code < 0x110000 else ""
    if len(name) == 1:
        return name
    return ""


class Font:
    def __init__(self, doc, d):
        self.doc = doc
        self.d = d or {}
        self.two_byte = False
        self.tounicode = {}
        self.diff = {}
        self.base_simple = True
        self._load()

    def _load(self):
        doc = self.doc
        d = self.d
        subtype = doc.resolve(d.get("Subtype"))
        if subtype == "Type0":
            self.two_byte = True
            self.base_simple = False
        tu = doc.resolve(d.get("ToUnicode"))
        if isinstance(tu, Stream):
            try:
                self.tounicode = parse_tounicode(tu.data())
            except Exception:
                self.tounicode = {}
        enc = doc.resolve(d.get("Encoding"))
        if isinstance(enc, dict):
            diffs = doc.resolve(enc.get("Differences")) or []
            code = 0
            for item in diffs:
                item = doc.resolve(item)
                if isinstance(item, (int, float)):
                    code = int(item)
                elif isinstance(item, str):
                    self.diff[code] = item
                    code += 1
        elif isinstance(enc, str) and enc in ("Identity-H", "Identity-V"):
            self.two_byte = True

    def codes(self, raw: bytes):
        if self.two_byte:
            return [int.from_bytes(raw[i:i + 2], "big") for i in range(0, len(raw) - 1, 2)]
        return list(raw)

    def decode(self, raw: bytes) -> str:
        out = []
        for c in self.codes(raw):
            if c in self.tounicode:
                out.append(self.tounicode[c])
                continue
            if c in self.diff:
                out.append(glyph_to_char(self.diff[c]))
                continue
            if self.two_byte:
                out.append("")
                continue
            if 32 <= c < 127:
                out.append(chr(c))
            elif c in STD_ENC_EXTRA:
                out.append(STD_ENC_EXTRA[c])
            elif c >= 160:
                out.append(bytes([c]).decode("latin-1"))
            else:
                out.append("")
        return "".join(out)


def page_text(doc: PDF, page: dict) -> str:
    res = doc.resolve(page.get("Resources")) or {}
    fonts_d = doc.resolve(res.get("Font")) or {}
    fonts = {}
    for name, ref in fonts_d.items():
        try:
            fonts[name] = Font(doc, doc.resolve(ref))
        except Exception:
            fonts[name] = Font(doc, {})
    contents = doc.resolve(page.get("Contents"))
    chunks = []
    if isinstance(contents, Stream):
        chunks.append(contents.data())
    elif isinstance(contents, list):
        for c in contents:
            c = doc.resolve(c)
            if isinstance(c, Stream):
                chunks.append(c.data())
    data = b"\n".join(chunks)
    return render_content(doc, data, fonts, res)


def render_content(doc, data, fonts, res, depth=0):
    lex = Parser(data, 0, doc)
    stack = []
    cur = None
    lines = []  # (y, x, text)
    tm = [1, 0, 0, 1, 0, 0]
    tlm = tm[:]
    fsize = 1.0
    out_parts = []
    cur_y = None
    cur_x = None
    buf = []

    def flush():
        nonlocal buf, cur_y, cur_x
        if buf:
            lines.append((cur_y, cur_x, "".join(buf)))
            buf = []

    def show(raw):
        nonlocal buf, cur_y, cur_x
        if cur is None:
            txt = raw.decode("latin-1", "replace")
        else:
            txt = cur.decode(raw)
        y = round(tm[5], 1)
        x = tm[4]
        if cur_y is None or abs(y - cur_y) > 0.6:
            flush()
            cur_y = y
            cur_x = x
        buf.append(txt)

    while True:
        save = lex.pos
        tok = lex.token()
        if tok is None:
            break
        kind, val = tok
        if kind != "kw":
            lex.pos = save
            try:
                stack.append(lex.parse())
            except Exception:
                lex.pos = save + 1
            if lex.pos <= save:
                lex.pos = save + 1
            continue
        op = val
        try:
            if op == "BT":
                tm = [1, 0, 0, 1, 0, 0]
                tlm = tm[:]
            elif op == "Tf":
                if len(stack) >= 2:
                    fname = stack[-2]
                    fsize = stack[-1] if isinstance(stack[-1], (int, float)) else 1.0
                    cur = fonts.get(fname)
            elif op == "Td":
                dx, dy = stack[-2], stack[-1]
                tlm = [tlm[0], tlm[1], tlm[2], tlm[3], tlm[4] + dx * tlm[0] + dy * tlm[2], tlm[5] + dx * tlm[1] + dy * tlm[3]]
                tm = tlm[:]
            elif op == "TD":
                dx, dy = stack[-2], stack[-1]
                tlm = [tlm[0], tlm[1], tlm[2], tlm[3], tlm[4] + dx * tlm[0] + dy * tlm[2], tlm[5] + dx * tlm[1] + dy * tlm[3]]
                tm = tlm[:]
            elif op == "Tm":
                tlm = [float(x) for x in stack[-6:]]
                tm = tlm[:]
            elif op in ("T*",):
                tlm = [tlm[0], tlm[1], tlm[2], tlm[3], tlm[4] - 12 * tlm[2], tlm[5] - 12 * tlm[3]]
                tm = tlm[:]
            elif op == "Tj":
                show(stack[-1])
            elif op == "'":
                tlm = [tlm[0], tlm[1], tlm[2], tlm[3], tlm[4], tlm[5] - 12]
                tm = tlm[:]
                show(stack[-1])
            elif op == '"':
                show(stack[-1])
            elif op == "TJ":
                arr = stack[-1]
                if isinstance(arr, list):
                    for it in arr:
                        if isinstance(it, bytes):
                            show(it)
                        elif isinstance(it, (int, float)) and it < -170:
                            if cur is not None or True:
                                buf.append(" ")
            elif op == "Do":
                name = stack[-1]
                xo = doc.resolve((doc.resolve(res.get("XObject")) or {}).get(name))
                if isinstance(xo, Stream) and doc.resolve(xo.dict.get("Subtype")) == "Form" and depth < 4:
                    sub_res = doc.resolve(xo.dict.get("Resources")) or res
                    sub_fonts = {}
                    for n2, r2 in (doc.resolve(sub_res.get("Font")) or {}).items():
                        sub_fonts[n2] = Font(doc, doc.resolve(r2))
                    flush()
                    lines.append((cur_y if cur_y is not None else 0, 0,
                                  render_content(doc, xo.data(), sub_fonts, sub_res, depth + 1)))
            elif op == "ET":
                flush()
        except (IndexError, TypeError, ValueError):
            pass
        if op not in ("Tf",):
            stack = []
        else:
            stack = []
    flush()
    lines.sort(key=lambda t: (-(t[0] or 0), t[1]))
    return "\n".join(t[2] for t in lines)


def main():
    path = sys.argv[1]
    lo = int(sys.argv[2])
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else lo
    doc = PDF(path)
    pages = doc.pages()
    for i in range(lo, hi + 1):
        if i - 1 >= len(pages):
            break
        print(f"\n===== PDF PAGE {i} =====")
        try:
            print(page_text(doc, pages[i - 1]))
        except Exception as exc:
            print(f"[error: {exc!r}]")


if __name__ == "__main__":
    main()
