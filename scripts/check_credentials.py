"""Credential smoke test: key presence and token HTTP status only; never prints values.

Usage: set -a; source .env; set +a; python scripts/check_credentials.py
"""

import os

import requests

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
KEYS = (
    "CDSE_USERNAME",
    "CDSE_PASSWORD",
    "SH_CLIENT_ID",
    "SH_CLIENT_SECRET",
    "OHSOME_API_KEY",
    "EE_PROJECT",
)


def token_status(data):
    response = requests.post(TOKEN_URL, data=data, timeout=(10, 30))
    ok = response.status_code == 200 and "access_token" in response.json()
    return f"HTTP {response.status_code} {'token OK' if ok else 'no token'}"


def main():
    for key in KEYS:
        print(f"{key}: {'set' if os.environ.get(key) else 'MISSING'}")
    if os.environ.get("SH_CLIENT_ID") and os.environ.get("SH_CLIENT_SECRET"):
        status = token_status(
            {
                "grant_type": "client_credentials",
                "client_id": os.environ["SH_CLIENT_ID"],
                "client_secret": os.environ["SH_CLIENT_SECRET"],
            }
        )
        print("Sentinel Hub OAuth client:", status)
    if os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD"):
        # One attempt only: repeated failures can lock the account.
        status = token_status(
            {
                "grant_type": "password",
                "client_id": "cdse-public",
                "username": os.environ["CDSE_USERNAME"],
                "password": os.environ["CDSE_PASSWORD"],
            }
        )
        print("CDSE account (password grant):", status)


if __name__ == "__main__":
    main()
