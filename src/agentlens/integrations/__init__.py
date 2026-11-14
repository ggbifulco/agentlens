"""Integrations with other tools.

- :mod:`agentlens.integrations.pytest_plugin`: run suites as pytest tests
  (``pytest --agentlens suites/x.yaml``) and the ``agentlens_run`` fixture (FR-S6).
- :mod:`agentlens.integrations.github`: post or update the report comment on a
  pull request (``agentlens github comment``, used by ``action.yml``; FR-C3).
- :mod:`agentlens.integrations.otel`: export traces as OpenTelemetry spans with
  the GenAI semantic conventions (FR-T6; needs ``agentlens-toolkit[otel]``).

This package is imported when pytest loads the plugin, so it must stay light:
no imports here.
"""
