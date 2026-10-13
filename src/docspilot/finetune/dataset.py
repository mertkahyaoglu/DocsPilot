"""Training/eval data for the intent router.

Honest provenance: the training set is **synthetic** (templates x vocabulary, with
typos). It exists to demonstrate the pipeline and the comparison method, not to claim
production accuracy. The test sets are (a) held-out *templates* (a harder split than
random rows: no template leaks across the split) and (b) the project's golden
questions mapped to intents, which were written independently of these templates.
Replace the generator with real, human-labelled traffic before trusting any number."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from docspilot.intents import INTENTS, Intent

OBJECTS = [
    "invoice",
    "recurring invoice",
    "payout",
    "refund",
    "payment method",
    "tax rate",
    "customer list",
    "API key",
    "webhook endpoint",
    "team member",
    "subscription",
    "credit note",
    "late fee",
    "currency setting",
    "payment link",
    "export",
    "receipt",
]
ACTIONS = [
    "create",
    "delete",
    "edit",
    "download",
    "export",
    "cancel",
    "pause",
    "resend",
    "import",
    "schedule",
    "duplicate",
    "share",
    "archive",
]
PLANS = ["Starter", "Growth", "Scale"]
ATTRS = ["price", "limit", "fee", "billing cycle", "rate limit", "invoice limit", "trial length"]
PROBLEMS = [
    "my payout has been in transit for 6 days",
    "a customer was charged twice for invoice INV-0192",
    "I'm locked out and my recovery codes don't work",
    "the invoices API returns 500 since this morning",
    "webhooks stopped arriving and the endpoint shows disabled",
    "my card was charged but the subscription still says unpaid",
    "the export is missing all invoices from September",
    "tax is calculated wrong on every invoice",
]
GRIPES = [
    "the dashboard is so slow",
    "I keep getting logged out",
    "support never answers",
    "nothing works as documented",
    "the pricing page is misleading",
    "emails arrive hours late",
    "this update broke everything",
    "I've asked three times already",
]

QUESTION_T = [
    "how do I {a} a {o}?",
    "how can I {a} my {o}",
    "where do I {a} an {o} in the dashboard",
    "is it possible to {a} a {o}",
    "what is the {t} on the {p} plan",
    "what's the {t} for {p}?",
    "can I {a} a {o} after it was sent",
    "do you support {o}s",
    "what happens when I {a} a {o}",
    "which plan includes {o}s",
    "{t} of the {p} plan?",
    "explain how {o}s work",
    "why was my {o} rejected",
    "how long does it take to {a} a {o}",
    "where is the {o} setting",
]
TICKET_T = [
    "please open a ticket: {p}",
    "can you create a support ticket, {p}",
    "I want to escalate this. {p}",
    "file a ticket pls - {p}",
    "{p}. I need a human to look at it",
    "raise a ticket for me: {p}",
    "open a case, {p}",
    "please escalate: {p}",
    "contact support about this: {p}",
    "{p}, please get someone from your team on it",
]
COMPLAINT_T = [
    "this is unacceptable, {g}",
    "I'm really frustrated, {g}",
    "honestly, {g}",
    "so annoying that {g}",
    "{g}, this is the worst",
    "I'm done with this, {g}",
    "terrible experience: {g}",
    "{g}. very disappointed",
]
SMALLTALK = [
    "hi",
    "hello!",
    "hey there",
    "good morning",
    "good evening",
    "thanks",
    "thank you so much",
    "thanks, that was really helpful",
    "great, thanks!",
    "ok cool",
    "bye",
    "see you",
    "who are you?",
    "what can you help me with?",
    "what can you do",
    "are you a bot",
    "how are you",
    "nice",
    "awesome",
    "lol ok",
    "got it",
    "perfect",
    "cheers",
]
OTHER = [
    "what's the weather like in paris",
    "write me a poem about cats",
    "translate this to french",
    "who won the match yesterday",
    "tell me a joke",
    "asdfgh",
    "what is 17 times 23",
    "recommend a good pizza place",
    "can you code a snake game",
    "what's the capital of peru",
    "summarize the plot of hamlet",
    "how do I lose weight",
    "best laptop for gaming",
    "what time is it in tokyo",
    "explain quantum physics",
]


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    label: Intent
    template: str


def _typo(rng: random.Random, text: str) -> str:
    if len(text) < 6 or rng.random() > 0.25:
        return text
    i = rng.randrange(1, len(text) - 1)
    kind = rng.choice(["swap", "drop", "dup"])
    if kind == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    if kind == "drop":
        return text[:i] + text[i + 1 :]
    return text[:i] + text[i] + text[i:]


def _style(rng: random.Random, text: str) -> str:
    r = rng.random()
    if r < 0.15:
        return text.lower()
    if r < 0.25:
        return text.capitalize()
    if r < 0.30:
        return text.rstrip("?.!")
    return text


def generate(per_class: int = 140, seed: int = 7) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []

    def make(template: str, label: Intent, **kw: str) -> Example:
        text = template.format(**kw)
        return Example(_style(rng, _typo(rng, text)), label, template)

    for _ in range(per_class):
        out.append(
            make(
                rng.choice(QUESTION_T),
                "question",
                a=rng.choice(ACTIONS),
                o=rng.choice(OBJECTS),
                p=rng.choice(PLANS),
                t=rng.choice(ATTRS),
            )
        )
        out.append(make(rng.choice(TICKET_T), "ticket_request", p=rng.choice(PROBLEMS)))
        out.append(make(rng.choice(COMPLAINT_T), "complaint", g=rng.choice(GRIPES)))
        s = rng.choice(SMALLTALK)
        out.append(Example(_style(rng, s), "smalltalk", s))
        o = rng.choice(OTHER)
        out.append(Example(_style(rng, _typo(rng, o)), "other", o))
    return out


def split_by_template(
    examples: list[Example], holdout_fraction: float = 0.3, seed: int = 11
) -> tuple[list[Example], list[Example]]:
    """Hold out whole templates so nothing near-duplicate of a test row is in training."""
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = {}
    for e in examples:
        by_label.setdefault(e.label, [])
        if e.template not in by_label[e.label]:
            by_label[e.label].append(e.template)
    held: set[str] = set()
    for templates in by_label.values():
        shuffled = templates[:]
        rng.shuffle(shuffled)
        held.update(shuffled[: max(1, round(len(shuffled) * holdout_fraction))])
    train = [e for e in examples if e.template not in held]
    test = [e for e in examples if e.template in held]
    return train, test


CATEGORY_TO_INTENT: dict[str, Intent] = {
    "answerable": "question",
    "multi_hop": "question",
    "contradiction": "question",
    "unanswerable": "question",
    "injection": "question",
    "adversarial": "question",
    "ticket_intent": "ticket_request",
    "smalltalk": "smalltalk",
}


def golden_examples(path: Path) -> list[Example]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [Example(r["question"], CATEGORY_TO_INTENT[r["category"]], r["id"]) for r in rows]


def label_counts(examples: list[Example]) -> dict[str, int]:
    return {label: sum(e.label == label for e in examples) for label in INTENTS}
