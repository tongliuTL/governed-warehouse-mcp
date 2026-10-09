"""Who is asking, and what they may see.

A principal is derived from a verified token. Access is expressed as OAuth scopes so the same
policy works for any identity provider that can mint JWTs:

========================  =============================================================
scope                     grants
========================  =============================================================
``warehouse:public``      public models (lanes, carriers, published marts)
``warehouse:internal``    internal models (shipments, customers, documents, quotes)
``warehouse:restricted``  restricted models (invoices, revenue, margin)
``pii:read``              PII columns (contact name, email, phone) on visible models
========================  =============================================================

Tier scopes are cumulative by design: a token with ``warehouse:restricted`` but not
``warehouse:internal`` still only sees what its *highest contiguous* tier allows, so a
mis-issued token can never skip a tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from warehouse_mcp.catalog import Catalog, Model, Tier

SCOPE_PUBLIC = "warehouse:public"
SCOPE_INTERNAL = "warehouse:internal"
SCOPE_RESTRICTED = "warehouse:restricted"
SCOPE_PII = "pii:read"

_TIER_SCOPES = (
    (Tier.PUBLIC, SCOPE_PUBLIC),
    (Tier.INTERNAL, SCOPE_INTERNAL),
    (Tier.RESTRICTED, SCOPE_RESTRICTED),
)

# Convenience presets for local development and the eval suite.
ROLE_SCOPES: dict[str, tuple[str, ...]] = {
    "partner": (SCOPE_PUBLIC,),
    "analyst": (SCOPE_PUBLIC, SCOPE_INTERNAL),
    "finance": (SCOPE_PUBLIC, SCOPE_INTERNAL, SCOPE_RESTRICTED),
    "support": (SCOPE_PUBLIC, SCOPE_INTERNAL, SCOPE_PII),
}


@dataclass(frozen=True)
class Principal:
    subject: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    client_id: str | None = None

    @classmethod
    def for_role(cls, role: str, subject: str | None = None) -> Principal:
        return cls(subject=subject or f"dev-{role}", scopes=frozenset(ROLE_SCOPES[role]), client_id="dev")

    @property
    def max_tier(self) -> Tier | None:
        """Highest tier reachable without skipping a lower one; None means no data access."""
        best: Tier | None = None
        for tier, scope in _TIER_SCOPES:
            if scope not in self.scopes:
                break
            best = tier
        return best

    @property
    def can_read_pii(self) -> bool:
        return SCOPE_PII in self.scopes

    def can_see(self, model: Model) -> bool:
        return self.can_see_tier(model.tier)

    def can_see_tier(self, tier: Tier) -> bool:
        return self.max_tier is not None and tier <= self.max_tier


class PolicyViolation(Exception):
    """A request the governance layer refuses. ``code`` is stable; ``message`` is agent-facing.

    ``internal_detail`` is recorded in the audit log only. It can hold facts the caller must
    not learn, such as the fact that a model they cannot see exists.
    """

    def __init__(self, code: str, message: str, *, internal_detail: str | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.internal_detail = internal_detail or message


def visible_model(catalog: Catalog, name: str, principal: Principal) -> Model:
    """The model, if the principal may see it. Unknown and invisible models get the same error,
    so a refusal never confirms that a restricted model exists."""
    model = catalog.by_name.get(name)
    if model is None or not principal.can_see(model):
        raise PolicyViolation(
            "UNKNOWN_MODEL",
            f"{name!r} is not a model you can query; use search_models to find one",
            internal_detail=None
            if model is None
            else f"{name} is {model.tier}; principal max tier is {principal.max_tier}",
        )
    return model
