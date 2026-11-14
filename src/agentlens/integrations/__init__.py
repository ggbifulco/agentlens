"""Integrations with other tools.

- :mod:`agentlens.integrations.pytest_plugin`: run suites as pytest tests
  (``pytest --agentlens suites/x.yaml``) and the ``agentlens_run`` fixture (FR-S6).

This package is imported when pytest loads the plugin, so it must stay light:
no imports here.
"""
