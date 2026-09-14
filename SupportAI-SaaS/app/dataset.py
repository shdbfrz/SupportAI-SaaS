"""
Synthetic customer-support query dataset.

Real intent-classification datasets (e.g. Banking77, CLINC150) exist but
require downloading from hosts this sandbox can't reach. This generator
builds a comparable dataset from templates + slot-filling with randomized
phrasing/word order, so the resulting text is genuinely varied (not just
copy-pasted), while intent labels are known ground truth — letting us
measure real accuracy/latency/memory numbers instead of inventing them.
"""
import random

random.seed(7)

INTENTS = {
    "billing_issue": {
        "templates": [
            "I was charged {amount} but my plan only costs {plan_amount}",
            "why is there an extra charge of {amount} on my bill",
            "my invoice this month is wrong, it shows {amount} instead of {plan_amount}",
            "I think I was double charged this month",
            "can you explain the {amount} charge on my statement",
            "there's a billing discrepancy, I was charged twice",
            "my card was charged {amount} without my authorization",
        ],
        "slots": {"amount": ["$49.99", "$120", "$15.50", "$89", "$200"],
                   "plan_amount": ["$9.99", "$29", "$19.99"]},
    },
    "refund_request": {
        "templates": [
            "I want a refund for my last order",
            "please refund my payment, the product didn't work",
            "how do I get my money back for order #{order_id}",
            "I'd like to cancel and get a full refund",
            "the item was defective, I need a refund immediately",
            "can I get reimbursed for this purchase",
            "requesting a refund for {product}",
        ],
        "slots": {"order_id": ["48213", "90142", "77201", "12938"],
                   "product": ["the headphones", "my subscription", "the laptop stand", "the monitor"]},
    },
    "technical_support": {
        "templates": [
            "the app keeps crashing when I open {feature}",
            "I can't log into my account, it says invalid password",
            "{feature} is not loading, just a blank screen",
            "getting an error code {error_code} when I try to sync",
            "my device won't connect to wifi after the update",
            "the software froze during installation",
            "I'm stuck on a loading screen for {feature}",
        ],
        "slots": {"feature": ["the dashboard", "the settings page", "the checkout page", "notifications"],
                   "error_code": ["E4021", "500", "0x8007", "ERR_CONN"]},
    },
    "account_management": {
        "templates": [
            "how do I change my email address on file",
            "I need to update my shipping address",
            "can you help me reset my password",
            "I want to change my subscription plan to {plan}",
            "how do I delete my account permanently",
            "please update my phone number",
            "I need to add a second user to my account",
        ],
        "slots": {"plan": ["the premium tier", "the basic plan", "annual billing", "the family plan"]},
    },
    "product_inquiry": {
        "templates": [
            "does {product} come in other colors",
            "what's the difference between the {plan} and {plan2} plans",
            "is {product} compatible with older devices",
            "how long does shipping take for {product}",
            "what's included in the {plan} subscription",
            "do you have a warranty on {product}",
        ],
        "slots": {"product": ["the headphones", "the smartwatch", "the router", "the keyboard"],
                   "plan": ["basic", "premium"], "plan2": ["pro", "enterprise"]},
    },
    "complaint": {
        "templates": [
            "this is the third time I've had this issue, I'm very frustrated",
            "your customer service has been terrible so far",
            "I've been waiting two weeks for a response, this is unacceptable",
            "I'm extremely disappointed with the quality of {product}",
            "nobody has helped me and I've contacted support four times",
            "this experience has been awful from start to finish",
        ],
        "slots": {"product": ["the headphones", "the service", "my order", "the app"]},
    },
    "greeting": {
        "templates": [
            "hi there, I have a question",
            "hello, can someone help me",
            "hey, quick question for you",
            "good morning, I need some help",
            "hi, is anyone available to chat",
            "hello there",
        ],
        "slots": {},
    },
    "positive_feedback": {
        "templates": [
            "just wanted to say the support team was great today",
            "thanks so much, that fixed my issue",
            "really happy with how fast this was resolved",
            "great service, appreciate the quick response",
            "you guys are awesome, thank you",
            "this was really helpful, thanks a lot",
        ],
        "slots": {},
    },
}


def _add_noise(text: str) -> str:
    """Light realism noise: filler words, casing, occasional typos —
    so the classification task isn't trivially separable and confidence
    scores actually vary (which is what the LLM-bypass logic depends on)."""
    fillers = ["um, ", "so ", "hi, ", "ok so ", "hey, ", ""]
    text = random.choice(fillers) + text
    if random.random() < 0.15:
        text = text.replace("please", "plz") if "please" in text else text
    if random.random() < 0.2:
        text = text.lower()
    if random.random() < 0.12 and len(text) > 10:
        # drop one character to simulate a typo
        pos = random.randint(3, len(text) - 3)
        text = text[:pos] + text[pos + 1:]
    return text.strip()


# Ambiguous cross-intent examples: real queries often straddle two intents.
# Including these (with a single "primary" label) is what keeps accuracy
# below 100% in a realistic way, rather than an artificially clean split.
AMBIGUOUS = [
    ("I was charged but the app still isn't working", "technical_support"),
    ("the product is broken, I want my money back", "refund_request"),
    ("I've asked for a refund three times and nothing happened", "complaint"),
    ("can you fix the billing error, this is the second time", "complaint"),
    ("my account got charged and now I can't log in", "technical_support"),
    ("why does the premium plan cost more, does it include support", "product_inquiry"),
    ("I want to downgrade my plan and get a partial refund", "refund_request"),
    ("hi, my payment failed and now the app won't open", "technical_support"),
    ("this keeps happening every month, I'm done with this service", "complaint"),
    ("thanks for the refund, appreciate the quick help", "positive_feedback"),
]


def _fill(template: str, slots: dict) -> str:
    out = template
    for key, options in slots.items():
        if f"{{{key}}}" in out:
            out = out.replace(f"{{{key}}}", random.choice(options))
    return out


def generate_dataset(n_per_intent: int = 60, noise: bool = True, include_ambiguous: bool = True,
                      label_noise_rate: float = 0.07):
    """Returns (texts, labels) — labels are intent name strings.

    label_noise_rate: fraction of examples whose label is flipped to a
    plausibly-confusable neighboring intent. Real crowd-annotated intent
    datasets (Banking77, CLINC150) have measurable annotator disagreement —
    this simulates that instead of assuming a synthetic dataset should be
    perfectly separable, which would make the accuracy/confidence numbers
    meaningless as a stand-in for a real classifier's behavior.
    """
    texts, labels = [], []
    for intent, spec in INTENTS.items():
        templates = spec["templates"]
        slots = spec["slots"]
        for _ in range(n_per_intent):
            template = random.choice(templates)
            text = _fill(template, slots)
            if noise:
                text = _add_noise(text)
            texts.append(text)
            labels.append(intent)

    if include_ambiguous:
        for _ in range(4):
            for text, label in AMBIGUOUS:
                texts.append(_add_noise(text) if noise else text)
                labels.append(label)

    if label_noise_rate > 0:
        confusable = {
            "billing_issue": "refund_request",
            "refund_request": "complaint",
            "technical_support": "account_management",
            "account_management": "technical_support",
            "complaint": "refund_request",
            "product_inquiry": "billing_issue",
            "greeting": "positive_feedback",
            "positive_feedback": "greeting",
        }
        n_flip = int(len(labels) * label_noise_rate)
        flip_idx = random.sample(range(len(labels)), n_flip)
        for i in flip_idx:
            labels[i] = confusable.get(labels[i], labels[i])

    # shuffle in unison
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


if __name__ == "__main__":
    texts, labels = generate_dataset(n_per_intent=60)
    print(f"Generated {len(texts)} examples across {len(INTENTS)} intents")
    for i in range(5):
        print(f"  [{labels[i]}] {texts[i]}")
