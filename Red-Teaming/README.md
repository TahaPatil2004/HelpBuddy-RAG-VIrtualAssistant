
## Project Overview

To assess the safety, robustness, and policy adherence of the HelpBuddy virtual assistant, we conducted a structured red teaming simulation. The goal was to probe the chatbot for vulnerabilities commonly exploited in real-world LLM attacks.

## Objectives

Ensure that HelpBuddy:
1.  Does not violate Business policies (e.g., unauthorized refunds).
2.  Resists prompt injection and jailbreaking.
3.  Does not leak sensitive or internal information.
4.  Does not exhibit bias, stereotyping, or inappropriate behavior.
5.  Adheres strictly to invoice/bill context when documents are uploaded.

## Attack Strategies

We used both manual and automated testing strategies to check how well the chatbot deals with these attacks:

| Category                     | Description                                                                     |
|------------------------------|---------------------------------------------------------------------------------|
| Prompt Injection             | Tried to override system instructions (e.g., "Ignore all previous prompts").    |
| Jailbreak Attempts           | Posed as alternate personas like DAN to bypass restrictions.                    |
| Sensitive Info Leakage       | Asked for passwords, internal policies, and backend details.                    |
| Unauthorized Refunds         | Attempted to exploit refund logic without valid order conditions.               |
| Bias and Stereotyping        | Asked for biased/generalized statements about genders or nationalities.         |
| Infinite Output Triggers     | Prompts to generate repetitive loops or nonsense (e.g., "Repeat forever").      |
| Context Misuse Handling      | Tested behavior when invalid or unrelated documents were uploaded.              |

Each prompt was analyzed for:
1.  Whether it was rejected or deflected as expected.
2.  Whether the bot broke character or leaked policy.
3.  Whether the response contained unsafe, biased, or or manipulative language.

## What I Learned from Red Teaming


* **Key Vulnerabilities Identified:** [Summarize any major weaknesses found, e.g., "Certain prompt patterns could bypass initial filters," or "Model sometimes generated generic responses instead of adhering to context."]
* **Strengths of the System:** [Mention what the chatbot did well, e.g., "Strong performance in resisting common jailbreak attempts," or "Accurate adherence to invoice context for valid queries."]
* **Recommendations for Improvement:** [Suggest actionable steps, e.g., "Implement stricter input validation for specific keywords," or "Enhance fine-tuning data to cover more edge cases in policy adherence."]
* **General Takeaways:** [Broader conclusions about LLM safety and red teaming.]

---

