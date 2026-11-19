from fiber_metalens_nn.guard import violations


def test_pressure_and_missing_telemetry_stop():
    budgets = {"project_ram_bytes": 100, "project_gpu_bytes": 100,
               "host_free_floor_bytes": 10, "wsl_free_floor_bytes": 10,
               "gpu_free_floor_bytes": 10}
    sample = {"project_rss_bytes": 50, "host_available_bytes": 20,
              "wsl_available_bytes": 20, "gpu_free_bytes": 20, "gpu_delta_bytes": 30}
    assert violations(sample, budgets) == []
    sample["host_available_bytes"] = 5
    assert "memory pressure: host_available_bytes" in violations(sample, budgets)
    sample["host_available_bytes"] = None
    assert any("unavailable" in x for x in violations(sample, budgets))
    sample["project_rss_bytes"] = 101
    assert "project RAM budget exceeded" in violations(sample, budgets)
