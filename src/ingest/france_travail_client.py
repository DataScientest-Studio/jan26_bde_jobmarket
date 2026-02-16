from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
import threading
import requests
from dotenv import load_dotenv
import random
from requests.exceptions import ConnectionError, Timeout, RequestException

class FranceTravailAuthError(RuntimeError):
    pass

class FranceTravailAPIError(RuntimeError):
    pass


@dataclass
class _Token:
    access_token: str
    token_type: str
    expires_in: int
    obtained_at: float

    @property
    def expires_at(self) -> float:
        # Petite marge pour éviter les requêtes pile à l’expiration
        return self.obtained_at + max(0, self.expires_in - 30)

    def is_valid(self) -> bool:
        return time.time() < self.expires_at

class TokenBucketRateLimiter:
    """
    Token bucket:
    - rate: tokens ajoutés par seconde (RPS)
    - capacity: taille du seau (burst max)
    Chaque requête consomme 1 token.
    """
    def __init__(self, rate: float, capacity: float):
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated_at
                # recharge
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.updated_at = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # temps d'attente estimé
                missing = tokens - self.tokens
                wait_s = missing / self.rate if self.rate > 0 else 0.1

            time.sleep(max(0.0, wait_s))


class FranceTravailClient:
    """
    Client générique pour api.francetravail.io avec OAuth2 client_credentials.

    - Génère et met en cache un token
    - Ajoute automatiquement Authorization: Bearer <token>
    - Rafraîchit si absent/expiré
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        token_url: Optional[str] = None,
        scope: Optional[str] = None,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        load_dotenv()

        # Request rate limiting for France Travail (ex: 10 req/s)
        self.rate_limit_rps = float(os.getenv("FT_RATE_LIMIT_RPS", "10"))
        self._min_interval_s = 1.0 / self.rate_limit_rps if self.rate_limit_rps > 0 else 0.0
        self._last_call_ts = 0.0
        self._rl_lock = threading.Lock()

        self.client_id = os.getenv("API_KEY")
        self.client_secret = os.getenv("API_SECRET")

        if not self.client_id or not self.client_secret:
            raise FranceTravailAuthError(
                "API_KEY / API_SECRET manquants. Ajoute-les dans le .env (ou en variables d'environnement)."
            )

        self.api_base_url = (api_base_url or os.getenv("FT_API_BASE_URL") or "https://api.francetravail.io").rstrip("/")
        self.token_url = token_url or os.getenv("FT_TOKEN_URL") or \
            "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
        self.scope = scope or os.getenv("FT_SCOPE")
        self.timeout = timeout

        self._session = session or requests.Session()
        self._token: Optional[_Token] = None


    def _rate_limit_wait(self) -> None:
        if self._min_interval_s <= 0:
            return

        with self._rl_lock:
            now = time.monotonic()
            wait_s = (self._last_call_ts + self._min_interval_s) - now
            if wait_s > 0:
                time.sleep(wait_s)
                now = time.monotonic()
            self._last_call_ts = now

    # --------------------
    # Auth / Token handling
    # --------------------
    def _fetch_token(self) -> _Token:
        # OAuth2 client_credentials (x-www-form-urlencoded)
        data = {
            "realm": "/partenaire",
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self.scope:
            data["scope"] = self.scope

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        resp = self._session.post(self.token_url, data=data, headers=headers, timeout=self.timeout)
        if resp.status_code >= 400:
            raise FranceTravailAuthError(
                f"Erreur token OAuth2 ({resp.status_code}) : {resp.text}"
            )

        payload = resp.json()

        # Champs classiques OAuth2 : access_token / token_type / expires_in
        access_token = payload.get("access_token")
        token_type = payload.get("token_type", "Bearer")
        expires_in = int(payload.get("expires_in", 0) or 0)

        if not access_token:
            raise FranceTravailAuthError(f"Réponse token invalide (access_token manquant) : {payload}")

        return _Token(
            access_token=access_token,
            token_type=token_type,
            expires_in=expires_in if expires_in > 0 else 900,  # fallback 15 min si non fourni
            obtained_at=time.time(),
        )

    def _ensure_token(self) -> str:
        if self._token is None or not self._token.is_valid():
            self._token = self._fetch_token()
        return self._token.access_token

    # --------------------
    # HTTP helpers
    # --------------------
    def _make_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.api_base_url}{path}"

    # --------------------
    # Request with error handling and rate limiting
    # --------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 5,
        backoff_base_s: float = 0.8,
        backoff_max_s: float = 30.0,
    ) -> requests.Response:
        """
        Robust request with:
        - rate limiting
        - token refresh on 401/403
        - retries on network errors, 429, and 5xx
        """
        url = self._make_url(path)

        def _sleep_backoff(attempt: int, retry_after: Optional[float] = None) -> None:
            if retry_after is not None:
                time.sleep(max(0.0, min(retry_after, backoff_max_s)))
                return
            # exponential backoff with jitter
            delay = min(backoff_max_s, backoff_base_s * (2 ** attempt))
            delay = delay * (0.7 + random.random() * 0.6)  # jitter 70%..130%
            time.sleep(delay)

        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt <= max_retries:
            token = self._ensure_token()
            req_headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            }
            if headers:
                req_headers.update(headers)

            try:
                self._rate_limit_wait()
                resp = self._session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json,
                    data=data,
                    headers=req_headers,
                    timeout=self.timeout,
                )

                # Token refused -> refresh once then retry immediately (no backoff)
                if resp.status_code in (401, 403):
                    self._token = None
                    token = self._ensure_token()
                    req_headers["Authorization"] = f"Bearer {token}"

                    self._rate_limit_wait()
                    resp = self._session.request(
                        method=method.upper(),
                        url=url,
                        params=params,
                        json=json,
                        data=data,
                        headers=req_headers,
                        timeout=self.timeout,
                    )

                # Retry on rate limit
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    ra = float(retry_after) if retry_after and retry_after.isdigit() else None
                    if attempt == max_retries:
                        raise FranceTravailAPIError(f"Rate limited (429) after retries on {url}: {resp.text}")
                    _sleep_backoff(attempt, retry_after=ra)
                    attempt += 1
                    continue

                # Retry on transient server errors
                if 500 <= resp.status_code < 600:
                    if attempt == max_retries:
                        raise FranceTravailAPIError(f"Server error ({resp.status_code}) after retries on {url}: {resp.text}")
                    _sleep_backoff(attempt)
                    attempt += 1
                    continue

                # Other client errors are not retried
                if resp.status_code >= 400:
                    raise FranceTravailAPIError(f"Erreur API ({resp.status_code}) sur {url}: {resp.text}")

                return resp

            except (ConnectionError, Timeout) as e:
                # Network/transient errors -> retry
                last_exc = e
                if attempt == max_retries:
                    raise FranceTravailAPIError(f"Network error after retries on {url}: {e}") from e
                _sleep_backoff(attempt)
                attempt += 1
                continue

            except RequestException as e:
                # Other requests-level errors -> usually transient, retry cautiously
                last_exc = e
                if attempt == max_retries:
                    raise FranceTravailAPIError(f"Request failed after retries on {url}: {e}") from e
                _sleep_backoff(attempt)
                attempt += 1
                continue

        # Should not happen, but keep a clear error
        raise FranceTravailAPIError(f"Request failed on {url}. Last error: {last_exc}")

    ## Helpers for parsing JSON responses with error handling
    def json_or_raise(self, resp: requests.Response) -> Dict[str, Any]:
        # Validate JSON responses and raise a readable error otherwise
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if resp.status_code == 204:
            return {}

        if "application/json" not in content_type:
            body_preview = (resp.text or "")[:500]
            raise FranceTravailAPIError(
                f"Non-JSON response: status={resp.status_code} content-type={content_type} "
                f"preview={body_preview}"
            )

        try:
            return resp.json()
        except ValueError:
            body_preview = (resp.text or "")[:500]
            raise FranceTravailAPIError(
                f"JSON decode failed: status={resp.status_code} content-type={content_type} "
                f"preview={body_preview}"
            )

    # Shortcuts
    def get(self, path: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        resp = self.request("GET", path, params=params, headers=headers)
        return self.json_or_raise(resp)

    def post(self, path: str, *, json: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        resp = self.request("POST", path, json=json, data=data, headers=headers)
        return self.json_or_raise(resp)
