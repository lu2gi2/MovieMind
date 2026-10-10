"""Small XML builders that reproduce the formatting patterns of the college's sample report.

Inline markup supported in text:  **bold**   __italic__   [[placeholder -> yellow highlight]]
"""
import re
from PIL import Image

TEXT_W = 9000  # table width used by the sample (dxa)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _run(txt, bold=False, italic=False, hl=False, sz=None, color=None, mono=False):
    rpr = ""
    if mono:
        rpr += '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
    if bold:
        rpr += "<w:b/><w:bCs/>"
    if italic:
        rpr += "<w:i/><w:iCs/>"
    if color:
        rpr += f'<w:color w:val="{color}"/>'
    if sz:
        rpr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    if hl:
        rpr += '<w:highlight w:val="yellow"/>'
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(txt)}</w:t></w:r>'


_TOK = re.compile(r"(\*\*.+?\*\*|__.+?__|\[\[.+?\]\]|`[^`]+`)", re.S)


def runs(text, sz=None, bold=False, italic=False, color=None):
    out = []
    for part in _TOK.split(text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(_run(part[2:-2], True, italic, False, sz, color))
        elif part.startswith("__"):
            out.append(_run(part[2:-2], bold, True, False, sz, color))
        elif part.startswith("`"):
            out.append(_run(part[1:-1], bold, italic, False, (sz or 24) - 2, color, mono=True))
        elif part.startswith("[["):
            out.append(_run("[" + part[2:-2] + "]", bold, italic, True, sz, color))
        else:
            out.append(_run(part, bold, italic, False, sz, color))
    return "".join(out)


# ---------------------------------------------------------------- paragraphs
def p_body(text, after=200, jc="both"):
    return f'<w:p><w:pPr><w:spacing w:after="{after}" w:line="360" w:lineRule="auto"/><w:jc w:val="{jc}"/></w:pPr>{runs(text)}</w:p>'


def p_italic(text):
    return f'<w:p><w:pPr><w:spacing w:after="80" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>{runs(text, italic=True)}</w:p>'


def p_bullet(text, num_id=1):
    return (f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>'
            f'<w:spacing w:after="120" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>{runs(text)}</w:p>')


def p_empty():
    return "<w:p/>"


def h1(chapter, title):
    rp = '<w:b/><w:bCs/><w:color w:val="000000" w:themeColor="text1"/>'
    big = rp + '<w:sz w:val="28"/><w:szCs w:val="28"/>'
    return ('<w:p><w:pPr><w:pStyle w:val="Heading1"/><w:pageBreakBefore/><w:spacing w:before="400" w:after="240"/><w:jc w:val="center"/>'
            f'<w:rPr>{rp}</w:rPr></w:pPr><w:r><w:rPr>{big}</w:rPr><w:t>{esc(chapter)}</w:t></w:r><w:r><w:rPr>{rp}</w:rPr><w:br/></w:r>'
            f'<w:r><w:rPr>{big}</w:rPr><w:t>{esc(title)}</w:t></w:r></w:p>')


def h2(text):
    return ('<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:keepNext/><w:spacing w:before="300" w:after="160"/><w:rPr><w:color w:val="000000" w:themeColor="text1"/></w:rPr></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="000000" w:themeColor="text1"/></w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def h3(text):
    return ('<w:p><w:pPr><w:pStyle w:val="Heading3"/><w:keepNext/><w:spacing w:before="200" w:after="120"/></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:bCs/><w:i/><w:iCs/><w:color w:val="000000" w:themeColor="text1"/></w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def caption(text, before=0):
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="{before}" w:after="120" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
            f'{runs(text, sz=22, bold=True)}</w:p>')


def caption_below(text):
    return (f'<w:p><w:pPr><w:spacing w:after="240" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
            f'{runs(text, sz=22, bold=True)}</w:p>')


def equation(text):
    return (f'<w:p><w:pPr><w:spacing w:before="80" w:after="200" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
            f'{_run(text, bold=True, italic=True)}</w:p>')


def code_block(text):
    out = []
    for ln in text.split("\n"):
        out.append('<w:p><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F5F5F5"/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
                   + (_run(ln if ln else " ", sz=16, mono=True)) + "</w:p>")
    return "".join(out) + '<w:p><w:pPr><w:spacing w:after="120"/></w:pPr></w:p>'


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


# ---------------------------------------------------------------- figures
_doc_pr = [200]


def figure(rid, png_path, width_in, cap):
    im = Image.open(png_path)
    w_emu = int(width_in * 914400)
    h_emu = int(w_emu * im.height / im.width)
    _doc_pr[0] += 1
    i = _doc_pr[0]
    draw = (f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{w_emu}" cy="{h_emu}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{i}" name="Picture {i}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name=""/><pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/></pic:cNvPicPr></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:srcRect/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{w_emu}" cy="{h_emu}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>')
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="80"/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr>{draw}</w:r></w:p>'
            + caption_below(cap))


# ---------------------------------------------------------------- tables
_MAR_H = '<w:tcMar><w:top w:w="100" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'
_MAR_B = '<w:tcMar><w:top w:w="80" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'


def table(headers, rows, widths, sz=22, header_fill="1F4E79", align=None):
    """Body table in the sample's navy-header style. widths in dxa; scaled to TEXT_W."""
    tot = sum(widths)
    widths = [int(w * TEXT_W / tot) for w in widths]
    align = align or ["left"] * len(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = ('<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="1F4E79"/>'
           '<w:left w:val="single" w:sz="4" w:space="0" w:color="1F4E79"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="1F4E79"/>'
           '<w:right w:val="single" w:sz="4" w:space="0" w:color="1F4E79"/><w:insideH w:val="single" w:sz="2" w:space="0" w:color="BFBFBF"/>'
           '<w:insideV w:val="single" w:sz="2" w:space="0" w:color="BFBFBF"/></w:tblBorders><w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
           '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr>'
           f"<w:tblGrid>{grid}</w:tblGrid>")
    out += "<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>"
    for h, w in zip(headers, widths):
        out += (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="{header_fill}"/>{_MAR_H}<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{runs(h, sz=sz, bold=True, color="FFFFFF")}</w:p></w:tc>')
    out += "</w:tr>"
    for i, r in enumerate(rows):
        out += "<w:tr><w:trPr><w:cantSplit/></w:trPr>"
        fill = '<w:shd w:val="clear" w:color="auto" w:fill="F4F8FC"/>' if i % 2 else ""
        for c, w, a in zip(r, widths, align):
            out += (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{fill}{_MAR_B}<w:vAlign w:val="center"/></w:tcPr>'
                    f'<w:p><w:pPr><w:jc w:val="{a}"/></w:pPr>{runs(str(c), sz=sz)}</w:p></w:tc>')
        out += "</w:tr>"
    return out + "</w:tbl>" + '<w:p><w:pPr><w:spacing w:after="160"/></w:pPr></w:p>'


def plain_table(headers, rows, widths, sz=22, align=None, total_w=9000, bold_first_col=False):
    """Black-bordered table used for the front-matter lists (Figures / Tables / Abbreviations)."""
    tot = sum(widths)
    widths = [int(w * total_w / tot) for w in widths]
    align = align or ["left"] * len(widths)
    b = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    borders = "".join(b.format(x) for x in ("top", "left", "bottom", "right", "insideH", "insideV"))
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = (f'<w:tbl><w:tblPr><w:tblW w:w="{total_w}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
           '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr>'
           f"<w:tblGrid>{grid}</w:tblGrid>")
    out += "<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>"
    for h, w in zip(headers, widths):
        out += (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="auto"/>{_MAR_H}<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{runs(h, sz=sz, bold=True)}</w:p></w:tc>')
    out += "</w:tr>"
    for r in rows:
        out += "<w:tr><w:trPr><w:cantSplit/></w:trPr>"
        for c, w, a in zip(r, widths, align):
            out += (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{_MAR_B}<w:vAlign w:val="center"/></w:tcPr>'
                    f'<w:p><w:pPr><w:jc w:val="{a}"/></w:pPr>{runs(str(c), sz=sz)}</w:p></w:tc>')
        out += "</w:tr>"
    return out + "</w:tbl>"


def toc_table(rows):
    """rows: (kind, no, title, page); kind in 'front','chap','sec'. Borderless, as in the sample."""
    widths = [1200, 6600, 1200]
    out = ('<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
           '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr>'
           + "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>")

    def cell(w, txt, jc="left", bold=False, ind=0):
        indp = f'<w:ind w:left="{ind}"/>' if ind else ""
        r = runs(txt, sz=22, bold=bold) if txt else ""
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr><w:p><w:pPr><w:spacing w:line="276" w:lineRule="auto"/>{indp}<w:jc w:val="{jc}"/></w:pPr>{r}</w:p></w:tc>')

    out += "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + cell(1200, "Ch. No.", "center", True) + cell(6600, "Title", "left", True) + cell(1200, "Page No.", "center", True) + "</w:tr>"
    for kind, no, t, pg in rows:
        out += ("<w:tr><w:trPr><w:cantSplit/></w:trPr>" + cell(1200, no, "center") + cell(6600, t, "left", False, 240 if kind == "sec" else 0)
                + cell(1200, str(pg) if pg not in (None, "") else "", "center") + "</w:tr>")
    return out + "</w:tbl>"
