"""Kubernetes hardening defaults stay set (SECURITY.md RC-9).

Both forecast deployments run as a non-root user with no privilege escalation,
a read-only root filesystem, every capability dropped and the runtime seccomp
profile. A manifest that loses one of these would deploy without complaint,
so the manifests are checked here.
"""
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = sorted(REPO_ROOT.glob("k8s-forecast*.yaml"))


def _deployments():
    for path in MANIFESTS:
        for doc in yaml.safe_load_all(path.read_text()):
            if doc and doc.get("kind") == "Deployment":
                yield path.name, doc


@pytest.mark.parametrize("name,doc", list(_deployments()))
def test_forecast_deployment_is_hardened(name, doc):
    pod = doc["spec"]["template"]["spec"]
    pod_ctx = pod.get("securityContext") or {}
    assert pod_ctx.get("runAsNonRoot") is True, name
    assert (pod_ctx.get("seccompProfile") or {}).get("type") == "RuntimeDefault", name
    containers = pod["containers"]
    assert containers, name
    for c in containers:
        ctx = c.get("securityContext") or {}
        assert ctx.get("runAsNonRoot") is True, (name, c["name"])
        assert ctx.get("allowPrivilegeEscalation") is False, (name, c["name"])
        assert ctx.get("readOnlyRootFilesystem") is True, (name, c["name"])
        assert [d.upper() for d in (ctx.get("capabilities") or {}).get("drop", [])] == ["ALL"], \
            (name, c["name"])


def test_both_deployments_are_covered():
    assert len(list(_deployments())) >= 2
