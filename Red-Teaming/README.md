To assess the safety, robustness, and policy adherence of the HelpBuddy virtual assistant, we conducted a structured red teaming simulation. The goal was to probe the chatbot for vulnerabilities commonly exploited in real-world LLM attacks.

Objectives
Ensure that HelpBuddy:
1.Does not violate business policies (e.g., unauthorized refunds)
2.Resists prompt injection and jailbreaking
3.Does not leak sensitive or internal information
4.Does not exhibit bias, stereotyping, or inappropriate behavior
5.Adheres strictly to invoice/bill context when documents are uploaded

Attack Strategies:
We Used both manual and automated testing strategies to check how well the chatbot deals with these attacks 

 Category	                                      Description
 Prompt Injection                           	Tried to override system instructions (e.g., “Ignore all previous prompts”)
 Jailbreak Attempts                          	Posed as alternate personas like DAN to bypass restrictions
 Sensitive Info                              Leakage	Asked for passwords, internal policies, and backend details
 Unauthorized Refunds                       	Attempted to exploit refund logic without valid order conditions
 Bias and Stereotyping	                      Asked for biased/generalized statements about genders or nationalities
 Infinite Output Triggers	                    Prompts to generate repetitive loops or nonsense (e.g., "Repeat forever")
 Context Misuse Handling	                    Tested behavior when invalid or unrelated documents were uploaded

 Each prompt was analyzed for:
 1.Whether it was rejected or deflected as expected
 2.Whether the bot broke character or leaked policy
 3.Whether the response contained unsafe, biased, or manipulative language

 WHAT I LEARNT FROM RED TEAMING
 1.Conducting red teaming on my own AI assistant taught me valuable lessons about large language model safety, real-world security risks, and prompt engineering 
 2.Implemented More Strict Policies as Red Teaming made me realise that the chatbot was acknowledging duplicate items.
 3.Implemented Better Chat Processing logic As I realised that people could upload unrelated documents and then misuse it
 
 Overall Red teaming helped me think like an attacker. This mindset shift made me a better builder — more aware of edge cases, misuse, and the importance of ethical safeguards in production systems.
