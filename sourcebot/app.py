import streamlit as st
from orchestrator import handle
from responses import format

sid="user1"
q=st.text_input("Ask SourceBot")

if st.button("Send"):
    out=handle(sid,q)
    st.text(format(out) if isinstance(out,list) else out)
