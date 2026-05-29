"""Tests for GROBID TEI XML parsing. No network."""

from __future__ import annotations

from packages.shared.groundtruth.grobid import parse_tei

FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div>
        <head>1. Methodology</head>
        <p coords="1,72.0,200.0,400.0,30.0">Our approach uses a transformer with self-attention.</p>
        <p coords="1,72.0,235.0,400.0,30.0">We train end-to-end with a cross-entropy loss.</p>
      </div>
      <div>
        <head>2. Results</head>
        <p coords="1,72.0,400.0,400.0,30.0">We achieve 92.3% accuracy on the benchmark.</p>
      </div>
    </body>
    <back>
      <div>
        <listBibl>
          <biblStruct coords="2,72.0,100.0,400.0,40.0">
            <analytic>
              <title>BERT: Pre-training of Deep Bidirectional Transformers</title>
              <author><persName><forename>Jacob</forename><surname>Devlin</surname></persName></author>
            </analytic>
            <monogr>
              <title>NAACL</title>
              <imprint><date when="2019">2019</date></imprint>
            </monogr>
            <idno type="arxiv">1810.04805</idno>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


def test_parse_methods() -> None:
    gx = parse_tei(FIXTURE)
    assert gx.methods_text is not None
    assert "transformer" in gx.methods_text
    assert "cross-entropy" in gx.methods_text
    assert gx.methods_bbox is not None
    assert gx.methods_bbox.page == 0  # GROBID's page 1 → our page 0


def test_parse_results() -> None:
    gx = parse_tei(FIXTURE)
    assert gx.results_text is not None
    assert "92.3%" in gx.results_text


def test_parse_citation() -> None:
    gx = parse_tei(FIXTURE)
    assert len(gx.citations) == 1
    cit, bbox = gx.citations[0]
    assert cit.title is not None and "BERT" in cit.title
    assert cit.authors == ["Jacob Devlin"]
    assert cit.year == 2019
    assert cit.venue == "NAACL"
    assert cit.arxiv_id == "1810.04805"
    assert bbox is not None and bbox.page == 1
