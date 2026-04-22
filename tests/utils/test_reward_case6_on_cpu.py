# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from examples.noise_math import reward_case_6


def _extra_info(question_text="Dummy question"):
    return {"prompt_text": f"system\nuser\nQuestion:\n{question_text}"}


def _ground_truth(answer="10"):
    return {"gold_answer": answer}


def _strict_response(target, steps, final_answer):
    return (
        f"[Goal Analysis]\n"
        f"Target: Var{{{target}}}\n"
        f"Plan: Solve for {target}.\n\n"
        f"[Backward Execution]\n"
        f"{steps}\n\n"
        f"[Final Answer]\n"
        f"{final_answer}"
    )


def test_step_rule_process_score_parse_failed_schema_is_stable():
    response = (
        "[Goal Analysis]\n"
        "Target: Var{X}\n"
        "Plan: test.\n\n"
        "[Backward Execution]\n"
        "This section has no valid strict numbered steps.\n\n"
        "[Final Answer]\n"
        "1"
    )
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info(),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
    )
    assert metrics["step_parse_failed"] == 1
    assert metrics["good_count_capped"] == 0
    assert metrics["step_constraint_mode"] == "failed"
    assert "step_details" in metrics


def test_compute_reward_step_rule_parse_failed_does_not_crash():
    response = (
        "[Goal Analysis]\n"
        "Target: Var{X}\n"
        "Plan: test.\n\n"
        "[Backward Execution]\n"
        "Still no strict step header here.\n\n"
        "[Final Answer]\n"
        "1"
    )
    result = reward_case_6.compute_reward(
        data_source="math_noise",
        solution_str=response,
        ground_truth=_ground_truth("1"),
        extra_info=_extra_info(),
        reward_mode="step_rule",
    )
    assert result["step_parse_failed"] == 1
    assert result["step_good_count_capped"] == 0
    assert "length_penalty" in result
    assert "step_parse_reason" in result


def test_good_step_cap_applies_to_strict_good_steps():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: First, use the number from the question.\n"
        "   [Source]: \"A is 2\"\n"
        "   [Calc]: Var{A} = <<2>>\n\n"
        "2. Define Var{B}:\n"
        "   [Reasoning]: Next, use Var{A} and the same question number 2 to derive B.\n"
        "   [Source]: Var{A} and \"2\"\n"
        "   [Calc]: Var{B} = <<2 + 2 = 4>>\n\n"
        "3. Define Var{C}:\n"
        "   [Reasoning]: Next, use Var{B} and the same question number 2 to derive C.\n"
        "   [Source]: Var{B} and \"2\"\n"
        "   [Calc]: Var{C} = <<4 + 2 = 6>>\n\n"
        "4. Define Var{D}:\n"
        "   [Reasoning]: Then, use Var{C} and the same question number 2 to derive D.\n"
        "   [Source]: Var{C} and \"2\"\n"
        "   [Calc]: Var{D} = <<6 + 2 = 8>>\n\n"
        "5. Calculate Var{Total}:\n"
        "   [Reasoning]: Finally, use Var{D} and the same question number 2 to get the target.\n"
        "   [Source]: Var{D} and \"2\"\n"
        "   [Calc]: Var{Total} = <<8 + 2 = 10>>"
    )
    response = _strict_response("Total", steps, "10")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("A is 2, and the chain may use 2, 4, 6, 8, and 10."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
    )
    assert metrics["good_count"] == 5
    assert metrics["good_count_capped"] == 3
    assert math.isclose(metrics["good_ratio"], 3 / 5)


def test_collapsed_multi_block_step_is_tracked_and_not_good():
    steps = (
        "1. Define Var{Total}:\n"
        "   [Reasoning]: First reasoning.\n"
        "   [Source]: \"Use 2\"\n"
        "   [Calc]: Var{Tmp} = <<2>>\n"
        "   [Reasoning]: Second reasoning.\n"
        "   [Source]: Var{Tmp}\n"
        "   [Calc]: Var{Total} = <<2 + 1 = 3>>"
    )
    response = _strict_response("Total", steps, "3")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("Use 2 and then add 1."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
    )
    assert metrics["collapsed_multi_block_count"] >= 1
    assert metrics["good_count"] == 0
    assert metrics["bad_count"] >= 1


def test_length_penalty_applies_after_four_steps():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: Set A.\n"
        "   [Source]: \"A is 1\"\n"
        "   [Calc]: Var{A} = <<1>>\n\n"
        "2. Define Var{B}:\n"
        "   [Reasoning]: Set B from A.\n"
        "   [Source]: Var{A}\n"
        "   [Calc]: Var{B} = <<Var{A} + 1 = 2>>\n\n"
        "3. Define Var{C}:\n"
        "   [Reasoning]: Set C from B.\n"
        "   [Source]: Var{B}\n"
        "   [Calc]: Var{C} = <<Var{B} + 1 = 3>>\n\n"
        "4. Define Var{D}:\n"
        "   [Reasoning]: Set D from C.\n"
        "   [Source]: Var{C}\n"
        "   [Calc]: Var{D} = <<Var{C} + 1 = 4>>\n\n"
        "5. Calculate Var{Total}:\n"
        "   [Reasoning]: Use D for the target.\n"
        "   [Source]: Var{D}\n"
        "   [Calc]: Var{Total} = <<Var{D} + 1 = 5>>"
    )
    response = _strict_response("Total", steps, "5")
    result = reward_case_6.compute_reward(
        data_source="math_noise",
        solution_str=response,
        ground_truth=_ground_truth("5"),
        extra_info=_extra_info("A is 1."),
        reward_mode="step_rule",
        length_penalty_start=4,
        length_penalty_per_step=0.1,
    )
    assert result["step_count"] == 5
    assert math.isclose(result["length_penalty"], 0.1)


def test_legacy_mode_returns_stable_schema():
    response = (
        "[Goal Analysis]\n"
        "Target: Var{X}\n"
        "Plan: test.\n\n"
        "[Backward Execution]\n"
        "1. Define Var{X}:\n"
        "   [Reasoning]: Simple.\n"
        "   [Source]: \"X is 1\"\n"
        "   [Calc]: Var{X} = <<1>>\n\n"
        "[Final Answer]\n"
        "1"
    )
    result = reward_case_6.compute_reward(
        data_source="math_noise",
        solution_str=response,
        ground_truth=_ground_truth("1"),
        extra_info=_extra_info(),
        reward_mode="legacy_overlap",
    )
    assert "step_good_count_capped" in result
    assert "length_penalty" in result
    assert "collapsed_multi_block_count" in result
    assert "structural_chain_count" in result
