from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests


SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_TRACKS_URL = "https://api.spotify.com/v1/tracks"


class SpotifyAuthError(Exception):
    pass


def _get_client_credentials() -> tuple[str, str]:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SpotifyAuthError("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in environment.")
    return client_id, client_secret


class SpotifyClient:
    def __init__(self) -> None:
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

    def _ensure_token(self) -> None:
        if self._access_token and time.time() < self._token_expiry - 30:
            return
        client_id, client_secret = _get_client_credentials()
        resp = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + int(data.get("expires_in", 3600))

    def fetch_tracks(self, spotify_ids: Iterable[str]) -> Dict[str, dict]:
        ids = [s for s in spotify_ids if s]
        result: Dict[str, dict] = {}
        if not ids:
            return result

        self._ensure_token()
        headers = {"Authorization": f"Bearer {self._access_token}"}
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i+batch_size]
            params = {"ids": ",".join(batch)}
            resp = requests.get(SPOTIFY_TRACKS_URL, headers=headers, params=params, timeout=15)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "1"))
                time.sleep(retry_after)
                resp = requests.get(SPOTIFY_TRACKS_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for track in data.get("tracks", []) or []:
                if track and track.get("id"):
                    result[track["id"]] = track
        return result


