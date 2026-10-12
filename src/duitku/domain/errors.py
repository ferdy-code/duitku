"""Minimal domain errors shared by repositories and (later) tools."""


class NotOwnedError(LookupError):
    """A referenced row (category, receipt, ...) does not exist for this user.

    Raised instead of returning data so a foreign id can never be attached to the caller's rows.
    """
