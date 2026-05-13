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
    assert metrics["bad_count_capped"] == 1
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


def test_case5_like_strict_good_steps_are_fully_rewarded():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: First, combine two grounded question numbers.\n"
        "   [Source]: \"Use 1 and 2\"\n"
        "   [Calc]: Var{A} = <<1 + 2 = 3>>\n\n"
        "2. Define Var{B}:\n"
        "   [Reasoning]: Next, use Var{A} and a grounded question number to derive B.\n"
        "   [Source]: Var{A} and \"3 and 2\"\n"
        "   [Calc]: Var{B} = <<3 + 2 = 5>>\n\n"
        "3. Define Var{C}:\n"
        "   [Reasoning]: Next, use Var{B} and another grounded question number to derive C.\n"
        "   [Source]: Var{B} and \"5 and 2\"\n"
        "   [Calc]: Var{C} = <<5 + 2 = 7>>\n\n"
        "4. Define Var{D}:\n"
        "   [Reasoning]: Then, use Var{C} and another grounded question number to derive D.\n"
        "   [Source]: Var{C} and \"7 and 2\"\n"
        "   [Calc]: Var{D} = <<7 + 2 = 9>>\n\n"
        "5. Calculate Var{Total}:\n"
        "   [Reasoning]: Finally, use Var{D} and a grounded question number to get the target.\n"
        "   [Source]: Var{D} and \"9 and 1\"\n"
        "   [Calc]: Var{Total} = <<9 + 1 = 10>>"
    )
    response = _strict_response("Total", steps, "10")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("Use 1 and 2, then plus 2, plus 2, plus 2, and finally plus 1."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
    )
    assert metrics["good_count"] >= 4
    assert metrics["good_count_capped"] == 3
    assert math.isclose(metrics["good_ratio"], 3 / 5)


def test_intermediate_step_with_calc_refs_can_be_good_without_future_refs():
    steps = (
        "1. Define Var{Base}:\n"
        "   [Reasoning]: Record the given base value.\n"
        "   [Source]: \"Base is 2\"\n"
        "   [Calc]: Var{Base} = <<2>>\n\n"
        "2. Define Var{Scratch}:\n"
        "   [Reasoning]: Derive a new grounded intermediate from Base.\n"
        "   [Source]: Var{Base} and \"plus 1\"\n"
        "   [Calc]: Var{Scratch} = <<Var{Base} + 1 = 3>><<2 + 1 = 3>>\n\n"
        "3. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target number from the question.\n"
        "   [Source]: \"Total is 5\"\n"
        "   [Calc]: Var{Total} = <<5>>"
    )
    response = _strict_response("Total", steps, "5")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("Base is 2. plus 1. Total is 5."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][1]["label"] == "good"


def test_direct_fact_copy_intermediate_is_neutral():
    steps = (
        "1. Define Var{Scratch}:\n"
        "   [Reasoning]: Record the question number as a grounded intermediate value.\n"
        "   [Source]: \"Scratch is 2\"\n"
        "   [Calc]: Var{Scratch} = <<2>>\n\n"
        "2. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target number from the question.\n"
        "   [Source]: \"Total is 5\"\n"
        "   [Calc]: Var{Total} = <<5>>"
    )
    response = _strict_response("Total", steps, "5")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("Scratch is 2. Total is 5."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][0]["label"] == "neutral"
    assert metrics["step_details"][0]["reason"] == "weak_contribution"


def test_intermediate_step_with_two_grounded_numbers_can_be_good():
    steps = (
        "1. Define Var{Combo}:\n"
        "   [Reasoning]: Combine two grounded question numbers into a new variable.\n"
        "   [Source]: \"There are 2 red marbles and 3 blue marbles\"\n"
        "   [Calc]: Var{Combo} = <<2 + 3 = 5>>\n\n"
        "2. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target number from the question.\n"
        "   [Source]: \"Total is 7\"\n"
        "   [Calc]: Var{Total} = <<7>>"
    )
    response = _strict_response("Total", steps, "7")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("There are 2 red marbles and 3 blue marbles. Total is 7."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][0]["label"] == "good"


def test_intermediate_step_with_operator_and_one_grounded_number_can_be_good():
    steps = (
        "1. Define Var{Base}:\n"
        "   [Reasoning]: Record the given base value.\n"
        "   [Source]: \"Base is 2\"\n"
        "   [Calc]: Var{Base} = <<2>>\n\n"
        "2. Define Var{Scaled}:\n"
        "   [Reasoning]: Scale the previous value using a simple arithmetic operator.\n"
        "   [Source]: Var{Base}\n"
        "   [Calc]: Var{Scaled} = <<Var{Base} * 2 = 4>><<2 * 2 = 4>>\n\n"
        "3. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target number from the question.\n"
        "   [Source]: \"Total is 7\"\n"
        "   [Calc]: Var{Total} = <<7>>"
    )
    response = _strict_response("Total", steps, "7")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("Base is 2. Total is 7."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][1]["label"] == "good"


def test_repeated_var_with_new_support_can_still_be_good():
    steps = (
        "1. Define Var{Base}:\n"
        "   [Reasoning]: Record the given base value.\n"
        "   [Source]: \"Base is 2\"\n"
        "   [Calc]: Var{Base} = <<2>>\n\n"
        "2. Define Var{Carry}:\n"
        "   [Reasoning]: Build an intermediate value from Base and a grounded number.\n"
        "   [Source]: Var{Base} and \"plus 3\"\n"
        "   [Calc]: Var{Carry} = <<Var{Base} + 3 = 5>><<2 + 3 = 5>>\n\n"
        "3. Define Var{Carry}:\n"
        "   [Reasoning]: Refine the same variable using new grounded support.\n"
        "   [Source]: Var{Base} and \"plus 4\"\n"
        "   [Calc]: Var{Carry} = <<Var{Base} + 4 = 6>><<2 + 4 = 6>>\n\n"
        "4. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target number from the question.\n"
        "   [Source]: \"Total is 6\"\n"
        "   [Calc]: Var{Total} = <<6>>"
    )
    response = _strict_response("Total", steps, "6")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("Base is 2. plus 3. plus 4. Total is 6."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][2]["label"] == "good"


def test_result_number_continuity_can_support_intermediate_good():
    steps = (
        "1. Define Var{Combo}:\n"
        "   [Reasoning]: Combine the grounded question numbers into an intermediate result.\n"
        "   [Source]: \"There are 2 red marbles and 3 blue marbles\"\n"
        "   [Calc]: Var{Combo} = <<2 + 3 = 5>>\n\n"
        "2. Define Var{Carry}:\n"
        "   [Reasoning]: Continue from the previous numeric result without naming the variable.\n"
        "   [Source]: \"Continue from above with 5 and add 1\"\n"
        "   [Calc]: Var{Carry} = <<5 + 1 = 6>>\n\n"
        "3. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target value.\n"
        "   [Source]: \"Total is 6\"\n"
        "   [Calc]: Var{Total} = <<6>>"
    )
    response = _strict_response("Total", steps, "6")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("There are 2 red marbles and 3 blue marbles. Total is 6."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][1]["label"] == "good"
    assert "result_number_continuity" in metrics["step_details"][1]["source_grounded_by"]


def test_intermediate_step_with_mixed_question_number_and_previous_var_can_be_good():
    steps = (
        "1. Define Var{Base}:\n"
        "   [Reasoning]: Record the base value.\n"
        "   [Source]: \"Base is 2\"\n"
        "   [Calc]: Var{Base} = <<2>>\n\n"
        "2. Define Var{Mixed}:\n"
        "   [Reasoning]: Combine the previous variable with a grounded question number.\n"
        "   [Source]: Var{Base} and \"plus 3\"\n"
        "   [Calc]: Var{Mixed} = <<2 + 3 = 5>>\n\n"
        "3. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target number from the question.\n"
        "   [Source]: \"Total is 8\"\n"
        "   [Calc]: Var{Total} = <<8>>"
    )
    response = _strict_response("Total", steps, "8")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("Base is 2 and plus 3. Total is 8."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][1]["label"] == "good"


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


def test_length_penalty_no_longer_affects_step_rule_score():
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
    assert math.isclose(result["length_penalty"], 0.0)
    expected = (
        0.7 * result["r_acc"]
        + 0.4 * result["gate_acc"] * (result["step_good_count_capped"] / result["step_norm_z"])
        - 0.3 * (result["step_bad_count_capped"] / result["step_norm_z"])
        + 0.2 * result["r_fmt"]
    )
    assert math.isclose(result["score"], expected)


def test_step_rule_uses_case5_like_formula_with_strict_format():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: Use the first number from the question.\n"
        "   [Source]: \"A is 2\"\n"
        "   [Calc]: Var{A} = <<2>>\n\n"
        "2. Define Var{B}:\n"
        "   [Reasoning]: Use Var{A} and the second number to derive B.\n"
        "   [Source]: Var{A} and \"B adds 3\"\n"
        "   [Calc]: Var{B} = <<2 + 3 = 5>>\n\n"
        "3. Calculate Var{Total}:\n"
        "   [Reasoning]: Use Var{B} to get the final target.\n"
        "   [Source]: Var{B}\n"
        "   [Calc]: Var{Total} = <<5>>"
    )
    response = _strict_response("Total", steps, "5")
    result = reward_case_6.compute_reward(
        data_source="math_noise",
        solution_str=response,
        ground_truth=_ground_truth("5"),
        extra_info=_extra_info("A is 2. B adds 3 more."),
        reward_mode="step_rule",
    )
    expected = (
        0.7 * result["r_acc"]
        + 0.4 * result["gate_acc"] * (result["step_good_count_capped"] / result["step_norm_z"])
        - 0.3 * (result["step_bad_count_capped"] / result["step_norm_z"])
        + 0.2 * result["r_fmt"]
    )
    assert math.isclose(result["score"], expected)
    assert math.isclose(result["length_penalty"], 0.0)
    assert result["r_fmt"] == 1.0
    assert math.isclose(result["gate_acc"], 1.0)


def test_wrong_answer_only_keeps_20_percent_of_good_process_signal():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: Use grounded question numbers to derive A.\n"
        "   [Source]: \"Use 2 and 3\"\n"
        "   [Calc]: Var{A} = <<2 + 3 = 5>>\n\n"
        "2. Define Var{B}:\n"
        "   [Reasoning]: Use Var{A} and a grounded question number to derive B.\n"
        "   [Source]: Var{A} and \"5 and 1\"\n"
        "   [Calc]: Var{B} = <<5 + 1 = 6>>\n\n"
        "3. Calculate Var{Total}:\n"
        "   [Reasoning]: Reuse Var{B} for the target variable.\n"
        "   [Source]: Var{B}\n"
        "   [Calc]: Var{Total} = <<6>>"
    )
    response = _strict_response("Total", steps, "999")
    result = reward_case_6.compute_reward(
        data_source="math_noise",
        solution_str=response,
        ground_truth=_ground_truth("6"),
        extra_info=_extra_info("Use 2 and 3, then add 1 to get the total."),
        reward_mode="step_rule",
    )
    good_ratio = result["step_good_count_capped"] / result["step_norm_z"]
    assert result["r_acc"] == 0.0
    assert result["step_good_count_capped"] > 0
    assert math.isclose(result["gate_acc"], 0.2)
    assert math.isclose(result["raw_good_term"], 0.4 * 0.2 * good_ratio)


def test_edge_case_reasons_only_keep_out_of_scope_neutral():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: Use a number that is outside the grounded source.\n"
        "   [Source]: \"A is 2\"\n"
        "   [Calc]: Var{A} = <<99>>\n\n"
        "2. Calculate Var{Total}:\n"
        "   [Reasoning]: First correct target step.\n"
        "   [Source]: \"Total is 1\"\n"
        "   [Calc]: Var{Total} = <<1>>\n\n"
        "3. Calculate Var{Total}:\n"
        "   [Reasoning]: Repeat the goal without any new support.\n"
        "   [Source]: \"Total stays 1\"\n"
        "   [Calc]: Var{Total} = <<1>>"
    )
    response = _strict_response("Total", steps, "1")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("A is 2. Total is 1."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    reasons = [detail["reason"] for detail in metrics["step_details"]]
    labels = [detail["label"] for detail in metrics["step_details"]]
    assert "out_of_scope" in reasons
    assert "duplicate_goal_without_new_dependency" in reasons
    assert labels[0] == "neutral"
    assert labels[2] == "bad"


def test_ungrounded_source_is_bad_when_source_grounding_is_required():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: Use a source sentence that does not ground to the question or previous vars.\n"
        "   [Source]: \"Some unrelated note\"\n"
        "   [Calc]: Var{A} = <<2>>\n\n"
        "2. Calculate Var{Total}:\n"
        "   [Reasoning]: First correct target step.\n"
        "   [Source]: \"Total is 1\"\n"
        "   [Calc]: Var{Total} = <<1>>"
    )
    response = _strict_response("Total", steps, "1")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("A starts from the number 2. Total is 1."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][0]["reason"] == "ungrounded_source"
    assert metrics["step_details"][0]["label"] == "bad"


def test_idle_chain_only_marks_clear_duplicate_repetition_bad():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: Set A.\n"
        "   [Source]: \"A is 2\"\n"
        "   [Calc]: Var{A} = <<2>>\n\n"
        "2. Define Var{A}:\n"
        "   [Reasoning]: Set A.\n"
        "   [Source]: \"A is 2\"\n"
        "   [Calc]: Var{A} = <<2>>\n\n"
        "3. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target number.\n"
        "   [Source]: \"Total is 5\"\n"
        "   [Calc]: Var{Total} = <<5>>"
    )
    response = _strict_response("Total", steps, "5")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("A is 2. Total is 5."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["step_details"][1]["reason"] == "idle_chain"
    assert metrics["step_details"][1]["label"] == "bad"


def test_good_and_bad_caps_are_applied_in_process_score():
    steps = (
        "1. Define Var{A}:\n"
        "   [Reasoning]: Combine two grounded ones.\n"
        "   [Source]: \"Use 1 and 1\"\n"
        "   [Calc]: Var{A} = <<1 + 1 = 2>>\n\n"
        "2. Define Var{B}:\n"
        "   [Reasoning]: Use Var{A} and a grounded question number.\n"
        "   [Source]: Var{A} and \"2 and 1\"\n"
        "   [Calc]: Var{B} = <<2 + 1 = 3>>\n\n"
        "3. Define Var{C}:\n"
        "   [Reasoning]: Use Var{B} and a grounded question number.\n"
        "   [Source]: Var{B} and \"3 and 1\"\n"
        "   [Calc]: Var{C} = <<3 + 1 = 4>>\n\n"
        "4. Define Var{D}:\n"
        "   [Reasoning]: Use Var{C} and a grounded question number.\n"
        "   [Source]: Var{C} and \"4 and 1\"\n"
        "   [Calc]: Var{D} = <<4 + 1 = 5>>\n\n"
        "5. Define Var{X}:\n"
        "   [Reasoning]: Repeat unsupported X.\n"
        "   [Source]: Var{Missing}\n"
        "   [Calc]: Var{X} = <<Var{Missing} + 1 = 2>>\n\n"
        "6. Define Var{Y}:\n"
        "   [Reasoning]: Repeat unsupported Y.\n"
        "   [Source]: Var{Missing}\n"
        "   [Calc]: Var{Y} = <<Var{Missing} + 1 = 2>>\n\n"
        "7. Define Var{Z}:\n"
        "   [Reasoning]: Repeat unsupported Z.\n"
        "   [Source]: Var{Missing}\n"
        "   [Calc]: Var{Z} = <<Var{Missing} + 1 = 2>>\n\n"
        "8. Define Var{W}:\n"
        "   [Reasoning]: Repeat unsupported W.\n"
        "   [Source]: Var{Missing}\n"
        "   [Calc]: Var{W} = <<Var{Missing} + 1 = 2>>\n\n"
        "9. Calculate Var{Total}:\n"
        "   [Reasoning]: Use the target number.\n"
        "   [Source]: Var{D} and \"5 and 4\"\n"
        "   [Calc]: Var{Total} = <<9>>"
    )
    response = _strict_response("Total", steps, "9")
    metrics = reward_case_6.compute_step_rule_process_score(
        response_str=response,
        extra_info=_extra_info("Use 1 and 1. plus 1. plus 1. plus 1. Total is 9."),
        step_norm_min=3,
        require_source_grounding=True,
        bad_on_duplicate_goal_without_new_dependency=True,
        bad_on_idle_chain=True,
        bad_on_invalid_dependency=True,
        step_parser_mode="strict",
        enable_natural_step_parser=False,
        good_step_cap=3,
        bad_step_cap=3,
    )
    assert metrics["good_count"] >= 4
    assert metrics["good_count_capped"] == 3
    assert metrics["bad_count"] >= 4
    assert metrics["bad_count_capped"] == 3


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
    assert "step_bad_count_capped" in result
    assert "length_penalty" in result
    assert "collapsed_multi_block_count" in result
    assert "structural_chain_count" in result
