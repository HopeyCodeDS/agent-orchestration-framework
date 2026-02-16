"""API key validation and multi-tenancy support.

Validates API keys from the X-API-Key header. All state-modifying routes
require authentication via this module.
"""

import hashlib
import hmac
from typing import Any

import structlog

log = structlog.get_logger()


class Tenant:
    """Represents an authenticated API tenant.

    Attributes:
        tenant_id: Unique tenant identifier.
        name: Display name for the tenant.
        tier: Subscription tier (free, standard, premium).
        quota_limit: Maximum requests per hour.
    """

    def __init__(
        self,
        tenant_id: str,
        name: str,
        tier: str = "free",
        quota_limit: int = 100,
    ) -> None:
        self.tenant_id = tenant_id
        self.name = name
        self.tier = tier
        self.quota_limit = quota_limit


class AuthService:
    """API key validation and tenant resolution.

    In production, this would validate against a secrets store.
    For development, uses an in-memory registry.
    """

    def __init__(self) -> None:
        self._api_keys: dict[str, Tenant] = {}
        self._usage: dict[str, int] = {}

    def register_api_key(self, api_key: str, tenant: Tenant) -> None:
        """Register an API key for a tenant.

        Args:
            api_key: The API key string.
            tenant: The tenant associated with this key.
        """
        key_hash = self._hash_key(api_key)
        self._api_keys[key_hash] = tenant

    async def validate_api_key(self, api_key: str) -> Tenant | None:
        """Validate an API key and return the associated tenant.

        Args:
            api_key: The API key from the request header.

        Returns:
            Tenant if valid, None if invalid.
        """
        if not api_key:
            return None

        key_hash = self._hash_key(api_key)
        tenant = self._api_keys.get(key_hash)

        if tenant is None:
            log.warning("auth.invalid_api_key")
            return None

        log.info("auth.validated", tenant_id=tenant.tenant_id)
        return tenant

    async def check_quota(self, tenant_id: str, quota_limit: int) -> bool:
        """Check if a tenant has remaining quota.

        Args:
            tenant_id: The tenant identifier.
            quota_limit: The maximum allowed requests.

        Returns:
            True if quota is available, False if exhausted.
        """
        current_usage = self._usage.get(tenant_id, 0)
        if current_usage >= quota_limit:
            log.warning(
                "auth.quota_exceeded",
                tenant_id=tenant_id,
                usage=current_usage,
                limit=quota_limit,
            )
            return False
        return True

    async def record_usage(self, tenant_id: str) -> None:
        """Record a usage event for quota tracking.

        Args:
            tenant_id: The tenant identifier.
        """
        self._usage[tenant_id] = self._usage.get(tenant_id, 0) + 1

    async def reset_usage(self, tenant_id: str) -> None:
        """Reset usage counter for a tenant (called hourly).

        Args:
            tenant_id: The tenant identifier.
        """
        self._usage[tenant_id] = 0

    def get_usage(self, tenant_id: str) -> dict[str, Any]:
        """Get usage stats for a tenant.

        Args:
            tenant_id: The tenant identifier.

        Returns:
            Dict with current usage information.
        """
        return {
            "tenant_id": tenant_id,
            "current_usage": self._usage.get(tenant_id, 0),
        }

    @staticmethod
    def _hash_key(api_key: str) -> str:
        """Hash an API key for secure storage/comparison.

        Args:
            api_key: The raw API key.

        Returns:
            SHA-256 hash of the key.
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def generate_api_key(tenant_id: str, secret: str = "dev-secret") -> str:
        """Generate an API key for a tenant (development use only).

        Args:
            tenant_id: The tenant identifier.
            secret: Secret key for HMAC generation.

        Returns:
            Generated API key string.
        """
        return hmac.new(
            secret.encode(),
            tenant_id.encode(),
            hashlib.sha256,
        ).hexdigest()
