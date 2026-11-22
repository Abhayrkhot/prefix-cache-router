from __future__ import annotations

import json

import pytest

from pcrouter.cli import build_parser, main


def test_simulate_compare_stats(tmp_path, capsys):
    base = ["--synthetic", "shared-prefix", "--n", "120", "--rate", "20"]
    cl = ["--workers", "2", "--kv-tokens", "200000"]
    assert main(["stats", *base]) == 0
    assert json.loads(capsys.readouterr().out)["requests"] == 120
    assert main(["simulate", *base, *cl, "--policy", "ttft", "--param", "beta=4"]) == 0
    assert "ttft" in capsys.readouterr().out
    assert main(["simulate", *base, *cl, "--policy", "chwbl", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["policy"] == "chwbl"
    out = tmp_path / "cmp.json"
    assert (
        main(
            [
                "compare",
                "--synthetic",
                "multiturn",
                "--n",
                "30",
                *cl,
                "--policies",
                "round-robin,affinity",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert set(json.loads(out.read_text())) == {"round-robin", "affinity"}


def test_parser_errors():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["simulate", "--synthetic", "multiturn", "--param", "nokv"])
    p = build_parser().parse_args(
        [
            "serve",
            "--worker",
            "http://a:1/",
            "--policy-param",
            "beta=2",
            "--policy-param",
            "mode=prefer",
        ]
    )
    from pcrouter.server.config import config_from_args

    cfg = config_from_args(p)
    assert cfg.workers == ["http://a:1"] and cfg.policy_params == {"beta": 2, "mode": "prefer"}
    with pytest.raises(SystemExit):
        config_from_args(build_parser().parse_args(["serve"]))
