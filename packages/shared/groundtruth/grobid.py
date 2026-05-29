"""GROBID client + TEI XML parser. Extracts methods, citations, results section."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from lxml import etree  # type: ignore[attr-defined]

from packages.extraction.schema import BBox, Citation

TEI_NS = {"t": "http://www.tei-c.org/ns/1.0"}
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"


def process(pdf_path: Path, base_url: str = GROBID_URL, timeout: float = 120.0) -> bytes:
    """POST PDF to GROBID, return TEI XML bytes."""
    with open(pdf_path, "rb") as f:
        files = {"input": (pdf_path.name, f, "application/pdf")}
        data = {
            "teiCoordinates": "head,p,s,biblStruct",
            "consolidateHeader": "0",
            "consolidateCitations": "0",
        }
        r = httpx.post(base_url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.content


def _bbox_union(s: str | None) -> BBox | None:
    """Parse GROBID coords ('page,x,y,w,h;...') into a union bbox on the first page."""
    if not s:
        return None
    parts = s.split(";")
    boxes = []
    for p in parts:
        bits = p.split(",")
        if len(bits) != 5:
            continue
        try:
            page = int(bits[0]) - 1
            x, y, w, h = (float(v) for v in bits[1:])
            boxes.append((page, x, y, x + w, y + h))
        except ValueError:
            continue
    if not boxes:
        return None
    page = boxes[0][0]
    same = [b for b in boxes if b[0] == page]
    return BBox(
        page=page,
        x0=min(b[1] for b in same),
        y0=min(b[2] for b in same),
        x1=max(b[3] for b in same),
        y1=max(b[4] for b in same),
    )


def _text(el: etree._Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


@dataclass
class GrobidExtraction:
    methods_text: str | None = None
    methods_bbox: BBox | None = None
    results_text: str | None = None
    results_bbox: BBox | None = None
    citations: list[tuple[Citation, BBox | None]] = field(default_factory=list)


_METHOD_HEAD = re.compile(
    r"\b(method|methodolog|approach|technique|architecture|"
    r"proposed|our (?:model|approach|method)|"
    r"experimental setup|setup)",
    re.IGNORECASE,
)
_RESULTS_HEAD = re.compile(r"\b(result|experiment|evaluation|finding)", re.IGNORECASE)
_INTRO_HEAD = re.compile(r"\b(introduction|background|related work|preliminaries)", re.IGNORECASE)


def _section_by_head(root: etree._Element, pattern: re.Pattern[str]) -> etree._Element | None:
    for div in root.xpath(".//t:body//t:div", namespaces=TEI_NS):
        head = div.find("t:head", TEI_NS)
        if head is not None and pattern.search(_text(head)):
            return div
    return None


def _fallback_methods_section(root: etree._Element) -> etree._Element | None:
    """First top-level body section whose head isn't intro/related/results."""
    for div in root.xpath("./t:text/t:body/t:div", namespaces=TEI_NS):
        head = div.find("t:head", TEI_NS)
        if head is None:
            continue
        text = _text(head)
        if _INTRO_HEAD.search(text) or _RESULTS_HEAD.search(text):
            continue
        return div
    return None


def _section_body(div: etree._Element) -> tuple[str, BBox | None]:
    paragraphs = div.findall("t:p", TEI_NS)
    text = " ".join(_text(p) for p in paragraphs).strip()
    coords = []
    for p in paragraphs:
        c = p.get("coords")
        if c:
            coords.append(c)
    bbox = _bbox_union(";".join(coords)) if coords else None
    return text, bbox


def _parse_citation(bs: etree._Element) -> Citation:
    title_el = bs.find(".//t:analytic/t:title", TEI_NS)
    if title_el is None:
        title_el = bs.find(".//t:monogr/t:title", TEI_NS)
    title = _text(title_el) if title_el is not None else None

    authors = []
    for pers in bs.findall(".//t:author/t:persName", TEI_NS):
        forename = pers.find("t:forename", TEI_NS)
        surname = pers.find("t:surname", TEI_NS)
        parts = []
        if forename is not None and forename.text:
            parts.append(forename.text)
        if surname is not None and surname.text:
            parts.append(surname.text)
        if parts:
            authors.append(" ".join(parts))

    year = None
    date_el = bs.find(".//t:date", TEI_NS)
    if date_el is not None:
        when = date_el.get("when") or (date_el.text or "")
        m = re.search(r"\d{4}", when)
        if m:
            year = int(m.group(0))

    venue_el = bs.find(".//t:monogr/t:title", TEI_NS)
    venue = _text(venue_el) if venue_el is not None and venue_el is not title_el else None

    doi = None
    arxiv_id = None
    for idno in bs.findall(".//t:idno", TEI_NS):
        kind = (idno.get("type") or "").lower()
        val = (idno.text or "").strip()
        if kind == "doi":
            doi = val
        elif kind == "arxiv":
            arxiv_id = val

    raw = _text(bs)
    return Citation(
        raw=raw, title=title, authors=authors, year=year, venue=venue,
        arxiv_id=arxiv_id, doi=doi,
    )


def parse_tei(tei_xml: bytes) -> GrobidExtraction:
    root = etree.fromstring(tei_xml)
    out = GrobidExtraction()

    methods_div = _section_by_head(root, _METHOD_HEAD) or _fallback_methods_section(root)
    if methods_div is not None:
        out.methods_text, out.methods_bbox = _section_body(methods_div)

    results_div = _section_by_head(root, _RESULTS_HEAD)
    if results_div is not None:
        out.results_text, out.results_bbox = _section_body(results_div)

    for bs in root.xpath(".//t:listBibl/t:biblStruct", namespaces=TEI_NS):
        cit = _parse_citation(bs)
        bbox = _bbox_union(bs.get("coords"))
        out.citations.append((cit, bbox))

    return out


def extract(pdf_path: Path, base_url: str = GROBID_URL) -> GrobidExtraction:
    tei = process(pdf_path, base_url=base_url)
    return parse_tei(tei)
