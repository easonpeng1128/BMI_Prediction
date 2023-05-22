import logging
import os

import streamlit as st
from twilio.rest import Client

logger = logging.getLogger(__name__)

os.environ["TWILIO_ACCOUNT_SID"] = "ACc49ce7b09c786c0e51665c98de32c904"
os.environ["TWILIO_AUTH_TOKEN"] = "723bac02ab626a4a499bf12dd4fa7396"


@st.cache_data
def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers
