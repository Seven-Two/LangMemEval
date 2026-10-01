"""Single source of truth for the LangMem/MemEval experiment adapter."""

def run(conv, llm_model, run_judge, category_names=None, judge_fn=None):
    return run_method("langmem", conv, llm_model, run_judge, category_names, judge_fn)


def run_method(method, conv, llm_model, run_judge, category_names=None, judge_fn=None):
    from openai import OpenAI
    from agents_memory.systems._helpers import _qa_results
    from agents_memory.token_tracker import phase
    from langmem_eval.benchmark import build_memory, select_context
    from langmem_eval.protocol import AnswerProtocol
    from langmem_eval.registry import create_backend
    from langmem_eval.model_api import answer_extra_options

    protocol = AnswerProtocol.from_env(judge_fn)
    model_options = answer_extra_options()
    with phase("write"):
        backend = create_backend(method, llm_model)
        build_memory(conv, backend)

    def answer(question):
        answer.trace = {"protocol": protocol.to_dict(), "messages": None,
                       "answer_called": False, "failure_stage": "retrieve"}
        if callable(getattr(backend, "describe", None)):
            answer.trace["method_config"] = backend.describe()
        with phase("retrieve"):
            records = backend.retrieve(question, protocol.top_k)[:protocol.top_k]
            selection = select_context(records, max_tokens=protocol.context_tokens,
                                       tokenizer=protocol.tokenizer)
        if getattr(backend, "last_retrieval", None) is not None:
            from copy import deepcopy
            answer.trace["retrieval_trace"] = deepcopy(backend.last_retrieval)
        answer.trace.update(selection.to_dict())
        if not selection.context and protocol.empty_context == "abstain":
            answer.trace.update(empty_context_action="abstain", failure_stage=None)
            return protocol.abstention_text
        messages = protocol.messages(question, selection.context)
        answer.trace.update(messages=messages, answer_called=True, failure_stage="answer",
                            empty_context_action="answer" if not selection.context else None)
        with phase("answer"):
            response = OpenAI().chat.completions.create(
                model=llm_model, temperature=protocol.temperature,
                max_tokens=protocol.max_output_tokens, messages=messages, **model_options,
            )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Answer model returned empty content")
        answer.trace.update(failure_stage=None,
                            finish_reason=getattr(response.choices[0], "finish_reason", None))
        return content

    return _qa_results(conv, answer, run_judge,
                       category_names=category_names, judge_fn=judge_fn)
