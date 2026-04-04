"""Unit tests for safety_tools (Pillar 1 — TNR)."""

from GraphRCA_agent.tools.safety_tools import (
    compute_health_score,
    health_regressed,
    UndoStack,
    UndoEntry,
    generate_revert_command,
    build_undo_entry,
)


class TestHealthScore:
    def test_zero_when_healthy(self):
        score = compute_health_score([], [], [])
        assert score == 0.0

    def test_weighted_sum(self):
        alerts = [1, 2]         # |A| = 2
        violations = [1]        # |V| = 1
        unhealthy = [1, 2, 3]  # |L| = 3
        # μ(s) = 0.4*2 + 0.35*1 + 0.25*3 = 0.8 + 0.35 + 0.75 = 1.90
        score = compute_health_score(alerts, violations, unhealthy)
        assert abs(score - 1.90) < 0.01

    def test_regression_detected(self):
        assert health_regressed(before=1.0, after=1.2, tolerance=0.05) is True

    def test_no_regression_within_tolerance(self):
        assert health_regressed(before=1.0, after=1.04, tolerance=0.05) is False

    def test_improvement_not_regression(self):
        assert health_regressed(before=2.0, after=0.5) is False


class TestUndoStack:
    def test_push_and_pop(self):
        stack = UndoStack()
        entry = UndoEntry("kubectl scale deploy/svc --replicas=5",
                          "kubectl scale deploy/svc --replicas=3",
                          "svc", "Scale up")
        stack.push(entry)
        assert len(stack) == 1
        popped = stack.pop()
        assert popped.action_cmd == "kubectl scale deploy/svc --replicas=5"
        assert len(stack) == 0

    def test_rollback_dry_run(self):
        stack = UndoStack()
        entry = UndoEntry("kubectl rollout restart deployment/svc",
                          "kubectl rollout undo deployment/svc",
                          "svc", "Restart")
        stack.push(entry)
        results = stack.rollback_all(dry_run=True)
        assert len(results) == 1
        assert results[0]["success"] is True

    def test_to_list_serializable(self):
        stack = UndoStack()
        entry = UndoEntry("cmd", "revert", "svc", "desc")
        stack.push(entry)
        data = stack.to_list()
        assert isinstance(data, list)
        assert data[0]["action_cmd"] == "cmd"


class TestGenerateRevertCommand:
    def test_scale_revert(self):
        cmd = "kubectl scale deployment/my-svc --replicas=5"
        revert = generate_revert_command(cmd)
        assert "my-svc" in revert
        assert "scale" in revert
        assert "--replicas=3" in revert  # 5 - 2 = 3

    def test_rollout_restart_revert(self):
        cmd = "kubectl rollout restart deployment/my-svc"
        revert = generate_revert_command(cmd)
        assert "undo" in revert
        assert "my-svc" in revert

    def test_set_env_revert_removes_keys(self):
        cmd = "kubectl set env deployment/svc FOO=bar BAZ=qux"
        revert = generate_revert_command(cmd)
        assert "FOO-" in revert
        assert "BAZ-" in revert

    def test_helm_upgrade_revert(self):
        cmd = "helm upgrade myrelease mychart --set x=1"
        revert = generate_revert_command(cmd)
        assert "rollback" in revert
        assert "myrelease" in revert

    def test_unknown_command_safe_noop(self):
        revert = generate_revert_command("some_custom_tool --flag")
        assert revert.startswith("#")


class TestBuildUndoEntry:
    def test_from_dict(self):
        action = {"command": "kubectl scale deployment/x --replicas=5",
                  "service": "x", "title": "Scale up"}
        entry = build_undo_entry(action)
        assert entry.service == "x"
        assert "x" in entry.revert_cmd

    def test_revert_auto_generated(self):
        action = {"command": "kubectl rollout restart deployment/nginx",
                  "service": "nginx", "title": "Restart nginx"}
        entry = build_undo_entry(action)
        assert "undo" in entry.revert_cmd
