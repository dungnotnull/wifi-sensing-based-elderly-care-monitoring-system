"""Internationalization manager for ElderCare alert messages.

Loads locale YAML files from configs/locales/ and provides template
interpolation via the ``t()`` helper. The active locale is read from
alerts.yaml (``language`` key, defaults to ``vi``).

Usage::

    from alerts.i18n import locale

    text = locale.t("alerts.fall_detected")
    text = locale.t("daily_summary.sleep_score_label", score=85, quality="good")
"""

import logging
import os
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_LOCALE = "vi"
_LOCALES_DIR = os.path.join("configs", "locales")


class LocaleManager:
    """Loads and caches locale dictionaries, provides ``t()`` look-up."""

    def __init__(
        self,
        locale_code: Optional[str] = None,
        locales_dir: Optional[str] = None,
    ) -> None:
        self._locales_dir = locales_dir or _LOCALES_DIR
        self._locale_code = locale_code or _DEFAULT_LOCALE
        self._catalog: dict[str, Any] = {}
        self._loaded: bool = False

    # -- public API ----------------------------------------------------------

    def configure(self, locale_code: Optional[str] = None, locales_dir: Optional[str] = None) -> None:
        """Reconfigure and force a reload on next access."""
        if locale_code is not None:
            self._locale_code = locale_code
        if locales_dir is not None:
            self._locales_dir = locales_dir
        self._loaded = False
        self._catalog = {}

    @property
    def locale_code(self) -> str:
        return self._locale_code

    def t(self, key: str, **kwargs: Any) -> str:
        """Look up *key* in the current locale catalog and interpolate.

        Nested keys are expressed with dot notation, e.g.
        ``"alerts.fall_detected"`` maps to ``catalog["alerts"]["fall_detected"]``.

        If the key is missing the raw dotted key string is returned so the
        application never crashes on a missing translation.
        """
        catalog = self._ensure_loaded()
        value = self._resolve(catalog, key)
        if value is None:
            logger.warning("Locale key not found: %s (locale=%s)", key, self._locale_code)
            return key
        if not isinstance(value, str):
            logger.warning("Locale key resolved to non-string: %s -> %r", key, value)
            return str(value)
        if not kwargs:
            return value
        try:
            return value.format(**kwargs)
        except KeyError as exc:
            logger.warning("Missing placeholder %s for key %s", exc, key)
            return value

    # -- internals -----------------------------------------------------------

    def _ensure_loaded(self) -> dict[str, Any]:
        if not self._loaded:
            self._catalog = self._load_yaml(self._locale_code)
            self._loaded = True
        return self._catalog

    def _load_yaml(self, locale_code: str) -> dict[str, Any]:
        path = os.path.join(self._locales_dir, f"{locale_code}.yaml")
        if not os.path.exists(path):
            logger.error("Locale file not found: %s", path)
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _resolve(catalog: dict[str, Any], dotted_key: str) -> Any:
        parts = dotted_key.split(".")
        node: Any = catalog
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node


def _init_locale_from_config() -> LocaleManager:
    """Factory: read language from alerts.yaml, fall back to ``vi``."""
    config_path = "configs/alerts.yaml"
    locale_code = _DEFAULT_LOCALE
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            if isinstance(cfg, dict):
                locale_code = cfg.get("language", _DEFAULT_LOCALE)
        except Exception:
            logger.exception("Failed to read locale from %s", config_path)
    return LocaleManager(locale_code=locale_code)


# Module-level singleton — import and use directly.
locale: LocaleManager = _init_locale_from_config()
