from __future__ import annotations

from run import parse_args


def test_parse_args_accepts_json_alias(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--json", "examples/custom_construct.json"],
    )
    args = parse_args()
    assert args.construct_file == "examples/custom_construct.json"


def test_parse_args_json_without_value_uses_default_file(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--json"],
    )
    args = parse_args()
    assert args.construct_file == "examples/custom_construct.json"

