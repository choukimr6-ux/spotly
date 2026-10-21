

def test_arabic_text_is_shaped_and_set_in_embedded_unicode_font():
    from kihc.analytics import certificate_authority as ca
    out = ca.safe_text("حملة رمضان")
    assert "?" not in out and ca.ARABIC_FONT in out and ca._arabic_support()
    assert ca.safe_text("Plain <b>") == "Plain &lt;b&gt;"
