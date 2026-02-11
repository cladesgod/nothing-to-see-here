"""API key authentication for the LM-AIG API.

Phase 1: Static API keys from environment variable (comma-separated).
Phase 2 (future): Database-backed key management with scopes.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class APIUser:
    """Authenticated API user."""

    user_id: str
    key_prefix: str  # First 8 chars of the key (for logging, not secret)


@dataclass
class APIKeyAuth:
    """API key authentication manager.

    Keys are stored as SHA-256 hashes for security. Even if the key store
    is compromised, original keys cannot be recovered.
    """

    _key_to_user: dict[str, APIUser] = field(default_factory=dict)

    def register_key(self, api_key: str, user_id: str) -> None:
        """Register an API key for a user."""
        key_hash = self._hash_key(api_key)
        prefix = api_key[:8] if len(api_key) >= 8 else api_key
        self._key_to_user[key_hash] = APIUser(user_id=user_id, key_prefix=prefix)

    def verify(self, api_key: str) -> APIUser | None:
        """Verify an API key and return the associated user.

        Uses constant-time comparison to prevent timing attacks.
        Returns None if the key is invalid.
        """
        if not api_key:
            return None
        key_hash = self._hash_key(api_key)
        for stored_hash, user in self._key_to_user.items():
            if hmac.compare_digest(key_hash, stored_hash):
                return user
        return None

    @staticmethod
    def generate_key() -> str:
        """Generate a cryptographically secure API key."""
        return f"aig_{secrets.token_urlsafe(32)}"

    @staticmethod
    def _hash_key(api_key: str) -> str:
        """Hash an API key with SHA-256."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    @classmethod
    def from_env_keys(cls, keys_csv: str) -> APIKeyAuth:
        """Create auth manager from comma-separated 'user_id:key' pairs.

        Format: "user1:key1,user2:key2" or just "key1,key2" (auto user IDs).
        """
        auth = cls()
        if not keys_csv.strip():
            return auth

        for i, entry in enumerate(keys_csv.split(",")):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                user_id, key = entry.split(":", 1)
            else:
                user_id = f"user_{i}"
                key = entry
            auth.register_key(key.strip(), user_id.strip())
            logger.info("api_key_registered", user_id=user_id.strip())

        return auth
