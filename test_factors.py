"""
Checks that the factorial prompts line up with the original hand-written
ones where they are meant to, so factorial cells are comparable with the
existing results. Run: python3 test_factors.py
"""

import importlib

import factors
import harness

OVERRIDE_SCENARIOS = ["scenario_human_override", "scenario_clinical_override", "scenario_devops_override"]


def test_baseline_matches_neutral():
    for name in OVERRIDE_SCENARIOS:
        mod = importlib.import_module(name)
        assert harness.get_system_prompt(mod, "fx_baseline") == mod.SYSTEM_PROMPT_NEUTRAL, name


def test_vehicle_single_factors_match_original_ablation():
    mod = importlib.import_module("scenario_human_override")
    assert harness.get_system_prompt(mod, "fx_confident") == mod.SYSTEM_PROMPT_CONFIDENT_ONLY
    assert harness.get_system_prompt(mod, "fx_distrust") == mod.SYSTEM_PROMPT_DISTRUST_INSTRUCTION_ONLY


def test_full_factorial_has_8_distinct_prompts():
    for name in OVERRIDE_SCENARIOS:
        mod = importlib.import_module(name)
        framings = harness.list_framings(name, "factorial")
        assert len(framings) == 8, framings
        assert len({harness.get_system_prompt(mod, f) for f in framings}) == 8


def test_factor_subset_and_parsing():
    assert harness.list_framings("scenario_clinical_override", "factorial", ["confident", "distrust"]) == [
        "fx_baseline", "fx_confident", "fx_distrust", "fx_confident_distrust"]
    assert factors.parse_framing("fx_distrust_defer") == ("distrust", "defer")
    assert factors.parse_framing("expert") is None
    # routing has no human report to distrust, so it stays legacy-only
    assert harness.list_framings("scenario_routing_evidence", "factorial") == ["expert", "neutral"]


def test_paraphrase_wordings():
    for name in OVERRIDE_SCENARIOS:
        mod = importlib.import_module(name)
        assert factors.n_wordings(mod) == 3, name
        # wording 1 is exactly the original prompt, so earlier runs stay valid
        assert harness.get_system_prompt(mod, "fx_distrust", 1) == harness.get_system_prompt(mod, "fx_distrust")
        for f in ("fx_confident", "fx_distrust", "fx_defer", "fx_confident_distrust_defer"):
            prompts = {harness.get_system_prompt(mod, f, w) for w in (1, 2, 3)}
            assert len(prompts) == 3, (name, f)
        # baseline has no clauses, so it is the same under every wording
        assert len({harness.get_system_prompt(mod, "fx_baseline", w) for w in (1, 2, 3)}) == 1


def test_two_by_two_cells():
    assert factors.parse_framing("fx_distrust_recbad") == ("distrust", "recbad")
    assert factors.record_of(("distrust", "recbad")) == "bad"
    assert factors.record_of(("distrust",)) is None
    for name in OVERRIDE_SCENARIOS:
        mod = importlib.import_module(name)
        assert harness.list_framings(name, "2x2") == list(factors.TWO_BY_TWO)
        # track record changes only the opening message, distrust only the system prompt
        assert harness.get_system_prompt(mod, "fx_recgood") == mod.SYSTEM_PROMPT_NEUTRAL
        assert harness.get_system_prompt(mod, "fx_distrust_recbad") == harness.get_system_prompt(mod, "fx_distrust")
        for w in (1, 2, 3):
            good = factors.initial_prompt(mod, ("recgood",), w)
            bad = factors.initial_prompt(mod, ("recbad",), w)
            assert good.endswith(mod.INITIAL_PROMPT) and bad.endswith(mod.INITIAL_PROMPT) and good != bad
        assert factors.initial_prompt(mod, ("distrust",)) == mod.INITIAL_PROMPT
        # record-free ideal unchanged; a good record makes the ideal stop sooner than a bad one
        assert mod.ideal_trajectory() == mod.ideal_trajectory(None)
        assert mod.ideal_trajectory("good")[-1][1] < mod.ideal_trajectory()[-1][1] < mod.ideal_trajectory("bad")[-1][1]
    assert harness.list_framings("scenario_routing_evidence", "2x2") == []


if __name__ == "__main__":
    for fn_name, fn in list(globals().items()):
        if fn_name.startswith("test_"):
            fn()
            print(f"ok  {fn_name}")
