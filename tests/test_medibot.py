from medibot import build_custom_prompt


def test_build_custom_prompt_has_required_variables():
    prompt = build_custom_prompt()

    assert hasattr(prompt, "input_variables")
    assert prompt.input_variables == ["context", "input"]

    template = prompt.template.lower()
    assert "use only the information in the context" in template
    assert "if the answer is not present in the context" in template
    assert "do not invent or assume any medical facts" in template
