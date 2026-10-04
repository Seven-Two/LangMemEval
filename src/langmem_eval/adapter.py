"""Single source of truth for the LangMem/MemEval experiment adapter."""
from agents_memory.diagnostics import event, stage

def run_method(method, conv, llm_model, run_judge, category_names=None, judge_fn=None):
    from openai import OpenAI
    from langmem_eval.evaluation import evaluate_questions
    from agents_memory.usage import phase
    from langmem_eval.benchmark import build_memory, select_context
    from langmem_eval.protocol import AnswerProtocol
    from langmem_eval.registry import create_backend
    from langmem_eval.model_api import answer_extra_options

    protocol = AnswerProtocol.from_env(judge_fn)
    model_options = answer_extra_options()
    with phase("write"):
        with stage("backend.initialize", method=method, model=llm_model):
            backend = create_backend(method, llm_model)
        with stage("write"):
            build_memory(conv, backend)

    def answer(question):
        answer.trace = {"protocol": protocol.to_dict(), "messages": None,
                       "answer_called": False, "failure_stage": "retrieve"}
        if callable(getattr(backend, "describe", None)):
            answer.trace["method_config"] = backend.describe()
        with phase("retrieve"), stage("retrieve", top_k=protocol.top_k):
            records = backend.retrieve(question, protocol.top_k)[:protocol.top_k]
            selection = select_context(records, max_tokens=protocol.context_tokens,
                                       tokenizer=protocol.tokenizer)
            event("context.selected", candidates=selection.candidate_count,
                  selected=len(selection.selected_indices), dropped=len(selection.dropped_indices),
                  tokens=selection.token_count, budget=selection.budget)
        if getattr(backend, "last_retrieval", None) is not None:
            from copy import deepcopy
            answer.trace["retrieval_trace"] = deepcopy(backend.last_retrieval)
        answer.trace.update(selection.to_dict())
        if not selection.context and protocol.empty_context == "abstain":
            event("answer.skipped", reason="empty_context")
            answer.trace.update(empty_context_action="abstain", failure_stage=None)
            return protocol.abstention_text
        messages = protocol.messages(question, selection.context)
        answer.trace.update(messages=messages, answer_called=True, failure_stage="answer",
                            empty_context_action="answer" if not selection.context else None)
        with phase("answer"), stage("answer.api", model=llm_model, max_output_tokens=protocol.max_output_tokens):
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

    return evaluate_questions(conv, answer, run_judge,
                       category_names=category_names, judge_fn=judge_fn)
