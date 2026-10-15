"""About and limits page: data, results, limitations and attributions (facts from REPORT.md)."""

import streamlit as st

st.markdown('<div class="kicker">About the system</div>', unsafe_allow_html=True)
st.title("About and limits")
st.write(
    "You give an area and a flood date. The system compares Sentinel-1 radar images from before "
    "and after the event, and adds Sentinel-2 when the sky is clear. It overlays OpenStreetMap from "
    "before the event, and finds settlements that can lose the road connection to the nearest town "
    "or hospital in four hypothetical scenarios. The copilot answers questions in English and Nepali "
    "from the same saved evidence. Every number comes from the maps."
)

left, right = st.columns(2)
with left:
    st.subheader("Measured results")
    st.markdown(
        """<div class="card"><b>Case study, Trishuli, 26 Aug 2026.</b> 22 of 73 settlements potentially
        road-cut in the primary scenario. 102 potentially exposed mapped buildings. 121 suspected affected
        road links.</div>
        <div class="card"><b>Flood segmentation model (UNet), development events.</b> Precision 0.8008,
        recall 0.9457, IoU 0.7656 (rule baseline IoU 0.6096).</div>
        <div class="card"><b>Copilot.</b> 62,263 evidence claims from the Trishuli run. Every answer cites its
        claims. A one-page report in English or Nepali takes about five seconds.</div>""",
        unsafe_allow_html=True,
    )
with right:
    st.subheader("How to read the evidence")
    st.markdown(
        """<div class="card">Overlap with a hazard means <b>potentially exposed</b>. Operational road status
        stays unknown until a field team reports it.</div>
        <div class="card">Areas that radar cannot see (layover and shadow in steep valleys) stay
        <b>unknown</b>, never zero.</div>
        <div class="card">Sentinel satellites revisit an area every few days. Satellite time is not event
        time.</div>
        <div class="card">The four access scenarios are <b>hypothetical</b>. They show which settlements depend
        on which links.</div>""",
        unsafe_allow_html=True,
    )

st.subheader("Attributions and citations")
st.markdown(
    """
- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30.
- © OpenStreetMap contributors.
- Kuro Siwo: Bountos et al., NeurIPS Datasets and Benchmarks 2024 (training data).
- Sen1Floods11: Bonafilia et al., CVPR Workshops 2020 (surveyed, not used).
- European Union, Copernicus Emergency Management Service data (EMSR927, validation only).
"""
)
st.caption("Full method and results: REPORT.pdf in the repository. Code: Apache License 2.0.")
